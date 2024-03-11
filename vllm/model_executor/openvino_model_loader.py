"""Utilities for selecting and loading models."""
import contextlib
import os
from typing import Optional

import math
import torch
import numpy as np
import torch.nn as nn

from vllm.config import DeviceConfig, ModelConfig
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput
from vllm.utils import is_openvino_optimum_intel

def flattenize_inputs(inputs):
    """
    Helper function for making nested inputs flattens
    """
    flatten_inputs = []
    for input_data in inputs:
        if input_data is None:
            continue
        if isinstance(input_data, (list, tuple)):
            flatten_inputs.extend(flattenize_inputs(input_data))
        elif isinstance(input_data, dict):
            flatten_inputs.extend(flattenize_inputs(list(input_data.values())))
        else:
            flatten_inputs.append(input_data)
    return flatten_inputs


def ov_wrapper(self, *args, **kwargs):
    #print('OV FORWARD WRAPPER')
    #print(f'model class: {type(args[0])}')
    #for i, input in enumerate(args[1:]):
    #    print(f'[{i}]: {type(input)}')
    #for key, value in kwargs.items():
    #    print(f'{key}: {type(value)}')
    #result = args[0]._openvino_patch_orig_forward(*args[1:], **kwargs)
    input_metadata = kwargs['input_metadata']
    #print(dir(input_metadata))
    #print(input_metadata.is_prompt, input_metadata.slot_mapping, input_metadata.max_context_len, input_metadata.context_lens, input_metadata.block_tables)
    def prepare_data(t):
        t = np.array(t, copy=False)
        #print(t.__array_interface__['data'][0])
        assert t.flags["C_CONTIGUOUS"]
        return t
    flatten_kv_cache = flattenize_inputs(kwargs['kv_caches'])
    #total_size = sum([torch.numel(t) for t in flatten_kv_cache])
    #print(f'kv-cache total size: {total_size}')
    flatten_kv_cache = [prepare_data(t) for t in flatten_kv_cache]
    inputs = [
        kwargs['input_ids'],
        kwargs['positions'],
        *flatten_kv_cache,
        input_metadata.is_prompt, input_metadata.slot_mapping
    ]
    #print('slot_mapping:', input_metadata.slot_mapping)
    if input_metadata.max_context_len is not None:
        # available from the second iteration
        inputs.append(input_metadata.max_context_len)
        inputs.append(input_metadata.context_lens)
        inputs.append(input_metadata.block_tables)
    else:
        inputs.append(np.array(0, dtype=np.int32))   # for optimum-based models this parameter can be used even on the first iteration
    #for input in inputs:
    #    print(f'{input.dtype} wiht shape {input.shape}' if isinstance(input, torch.Tensor) else type(input))
    result = self.ov_request.infer(inputs, share_inputs=True, share_outputs=False)
    #print(f'result: {type(result)}')
    return torch.from_numpy(result[0])

def patch_stateful_model(model, factory):
    print('TRANSFORMING OPTIMUM-INTEL MODEL TO vLLM COMPATIBLE FORM')
    from openvino.runtime.passes import Manager, MatcherPass, WrapType, Matcher, AnyInput, Or
    from openvino.runtime import opset13
    from openvino.runtime.utils import replace_node

    #model.remove_parameter(model.input('beam_idx').get_node())
    max_context_len = opset13.parameter(shape=[], dtype=np.int64, name='max_context_len')  # max_context_len
    model_remaining_params = [
        opset13.parameter(shape=[], dtype=bool, name='is_prompt'),  # is_prompt
        opset13.parameter(shape=[-1, -1], dtype=np.int64, name='slot_mapping'),  # slot mapping
        max_context_len,
        opset13.parameter(shape=[-1], dtype=np.int64, name='context_lens'),  # context_lens
        opset13.parameter(shape=[-1, -1], dtype=np.int32, name='block_tables'),  # block_tables
    ]
    for parameter in model_remaining_params:
        parameter.get_output_tensor(0).set_names({parameter.get_friendly_name()})
    paged_attention_remaining_args = [
        opset13.constant(np.array([], np.float32)),  # alibi_slopes
        opset13.constant(np.array(0, np.int32)),  # sliding_window
    ]

    kv_parameters = []
    assignes_to_remove = []
    parameters_to_remove = []
    results_to_remove = []
    position_ids_parameter = []

    class StateManagementPattern(MatcherPass):
        def __init__(self):
            MatcherPass.__init__(self)
            self.model_changed = False

            k_past_var = WrapType("opset13.ReadValue", AnyInput())
            k_past_par = WrapType("opset13.Parameter")
            k_past = Or([WrapType("opset13.Gather", [k_past_var, AnyInput(), AnyInput()]), k_past_par])
            k_current = AnyInput()
            k_concat = WrapType("opset13.Concat", [k_past, k_current])

            def kv_shaping(kv_concat):
                interim = WrapType("opset13.StridedSlice", [kv_concat, *[AnyInput() for _ in range(3)]])
                interim = WrapType("opset13.StridedSlice", [interim, *[AnyInput() for _ in range(3)]])
                unsqueeze = WrapType("opset13.Unsqueeze", [Or([kv_concat, interim]), AnyInput()])
                interim = WrapType("opset13.StridedSlice", [unsqueeze, *[AnyInput() for _ in range(3)]])
                interim = WrapType("opset13.StridedSlice", [interim, *[AnyInput() for _ in range(3)]])
                interim = WrapType("opset13.Broadcast", [Or([unsqueeze, interim]), AnyInput()])
                interim = WrapType("opset13.Reshape", [interim, AnyInput()])
                return interim

            v_past_var = WrapType("opset13.ReadValue", AnyInput())
            v_past_par = WrapType("opset13.Parameter")
            v_past = Or([WrapType("opset13.Gather", [v_past_var, AnyInput(), AnyInput()]), v_past_par])
            v_current = AnyInput()
            v_concat = WrapType("opset13.Concat", [v_past, v_current])

            q = AnyInput()
            sdpa = WrapType("opset13.ScaledDotProductAttention", [
                q,
                Or([k_concat, kv_shaping(k_concat)]),
                Or([v_concat, kv_shaping(v_concat)]),
                AnyInput()
            ])

            def callback(m: Matcher) -> bool:
                assert sdpa in m.get_pattern_value_map()
                mapping = m.get_pattern_value_map()
                assert sdpa in mapping
                real_q = mapping[q]
                real_k = mapping[k_current]
                real_v = mapping[v_current]
                hidden_shape = real_q.get_partial_shape()
                hidden_dim = hidden_shape[hidden_shape.rank.get_length() - 1].get_length()  # TODO: What if it is a dynamic? Need to insert a ShapeOf sub-graph instead
                k_parameter = opset13.parameter(shape=[-1, -1, -1, -1, -1], dtype=np.float32)
                v_parameter = opset13.parameter(shape=[-1, -1, -1, -1], dtype=np.float32)
                kv_parameters.append(k_parameter)
                kv_parameters.append(v_parameter)
                # TODO: The rank 4 is used in the following code, but it is not guaranteed for all models, adopt to other ranks.
                q_transpose = opset13.transpose(real_q, opset13.constant([0, 2, 1, 3]))
                q_reshape = opset13.reshape(q_transpose, opset13.constant([0, 0, -1]), True)
                k_transpose = opset13.transpose(real_k, opset13.constant([0, 2, 1, 3]))
                k_reshape = opset13.reshape(k_transpose, opset13.constant([0, 0, -1]), True)
                v_transpose = opset13.transpose(real_v, opset13.constant([0, 2, 1, 3]))
                v_reshape = opset13.reshape(v_transpose, opset13.constant([0, 0, -1]), True)
                # TODO: Detect whether SDPA in the model graph has scale argument set and use it instead of the computed scale below
                scale = opset13.constant(np.array(1.0/math.sqrt(float(hidden_dim)), dtype=np.float32))
                paged_attention = factory.create("PagedAttentionExtension", [
                    q_reshape,
                    k_reshape,
                    v_reshape,
                    k_parameter,
                    v_parameter,
                    *model_remaining_params,
                    scale,
                    *paged_attention_remaining_args
                ])
                pa_reshape = opset13.reshape(paged_attention, [0, 0, -1, hidden_dim], True)
                pa_transpose = opset13.transpose(pa_reshape, opset13.constant([0, 2, 1, 3]))

                # def add_kv_parameter(past_node):
                #     if past_node.get_type_info().name == 'Parameter':
                #         parameters_to_remove.append(past_node)

                # add_kv_parameter(mapping[k_gather])
                # add_kv_parameter(mapping[v_gather])

                if v_past_par in mapping:
                    parameters_to_remove.append(mapping[v_past_par].get_node())

                if k_past_par in mapping:
                    parameters_to_remove.append(mapping[k_past_par].get_node())

                def add_assign_consumers(output):
                    for consumer in output.get_target_inputs():
                        consumer_node = consumer.get_node()
                        consumer_type = consumer_node.get_type_info().name
                        if consumer_type == 'Assign':  # stateful model
                            assignes_to_remove.append(consumer_node)
                        elif consumer_type == 'Result':  # stateless model
                            results_to_remove.append(consumer_node)

                add_assign_consumers(mapping[k_concat])
                add_assign_consumers(mapping[v_concat])

                replace_node(m.get_match_root(), pa_transpose)
                print('INSERTED PageAttentionExtension')
                return True

            self.register_matcher(Matcher(sdpa, "StateAndSDPA"), callback)

    class MaxSequenceLengthPattern(MatcherPass):
        def __init__(self):
            MatcherPass.__init__(self)
            self.model_changed = False

            kv_past = WrapType("opset13.ReadValue", AnyInput())
            kv_gather = WrapType("opset13.Gather", [kv_past, AnyInput(), AnyInput()])
            kv_shape = WrapType("opset13.ShapeOf", [kv_gather])
            seq = WrapType("opset13.Gather", [kv_shape, AnyInput(), AnyInput()])

            def callback(m: Matcher) -> bool:
                replace_node(m.get_match_root(), max_context_len)
                print("DETECTED PATTERN FOR max_sequence_length, CONNECTED TO A DEDICATED PARAMETER")
                return True

            self.register_matcher(Matcher(seq, "MaxSequenceLengthPattern"), callback)

    # TODO: Instead of using the following transformation that matches quite a specific place in a model graph in case when position_ids parameter is missing,
    #       consider replacing always existing attention_mask parameter with a sub-graph using a new slot_mapping parameter.
    class PositionIDsReplacer(MatcherPass):
        def __init__(self):
            MatcherPass.__init__(self)
            self.model_changed = False

            input_ids = AnyInput()
            input_embed = WrapType("opset13.Gather", [AnyInput(), input_ids, AnyInput()])

            position_ids = AnyInput()
            offset = WrapType('opset13.Constant')
            add_offset = WrapType('opset13.Add', [position_ids, offset])
            convert = WrapType('opset13.Convert', [add_offset])
            position_embed = WrapType("opset13.Gather", [AnyInput(), convert, AnyInput()])

            add = WrapType("opset13.Add", [input_embed, position_embed])

            def callback(m: Matcher) -> bool:
                mapping = m.get_pattern_value_map()
                if not position_ids_parameter:
                    position_ids_parameter.append(opset13.parameter(shape=[-1, -1], dtype=np.int64, name="position_ids"))
                    print('CREATED A NEW position_ids PARAMETER')
                replace_node(mapping[position_ids].get_node(), position_ids_parameter[0])
                position_ids_parameter[0].get_output_tensor(0).set_names({'position_ids'})
                print('APPLIED position_ids PARAMETER INSTEAD OF attention_mask-BASED SUB-GRAPH')
                return True

            self.register_matcher(Matcher(add, "InputAndPoistionIDsAdd"), callback)

    m = Manager()
    m.set_per_pass_validation(False)
    m.register_pass(StateManagementPattern())
    m.register_pass(MaxSequenceLengthPattern())

    def has_parameter(model, name):
        return name in sum([list(t.get_names()) for t in model.inputs], [])

    if has_parameter(model, 'position_ids'):
        position_ids_parameter.append(model.input('position_ids').get_node())
    else:
        m.register_pass(PositionIDsReplacer())

    m.run_passes(model)

    if has_parameter(model, 'beam_idx'):
        model.remove_parameter(model.input('beam_idx').get_node())
    model.remove_parameter(model.input('attention_mask').get_node())
    # print('parameters_to_remove:', parameters_to_remove)
    # print('results_to_remove:', results_to_remove)
    # print('sinks_to_remove:', assignes_to_remove)
    for parameter in parameters_to_remove:
        model.remove_parameter(parameter)
    for sink in assignes_to_remove:
        model.remove_sink(sink)
    for result in results_to_remove:
        model.remove_result(result)
    if not has_parameter(model, 'position_ids'):
        model.add_parameters(position_ids_parameter)
    model.add_parameters(kv_parameters)
    model.add_parameters(model_remaining_params)
    print('PARAMETERS ARE REORGANIZED, THE STATE (IF EXISTS) IS REMOVED')

def ov_sample(
    self,
    hidden_states: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> Optional[SamplerOutput]:
    return self.sampler(None, hidden_states, sampling_metadata)

def get_model(model_config: ModelConfig, device_config: DeviceConfig,
              **kwargs) -> nn.Module:
    if not is_openvino_optimum_intel():
        from vllm.model_executor.model_loader import get_model
        return get_model(model_config, device_config, **kwargs)

    lora_config = kwargs.get("lora_config", None)
    if lora_config:
        raise ValueError(
            f"OpenVINO modeling does not support LoRA, "
            "but LoRA is enabled. Support for this model may "
            "be added in the future. If this is important to you, "
            "please open an issue on github.")

    import openvino as ov
    from optimum.intel import OVModelForCausalLM
    pt_model = OVModelForCausalLM.from_pretrained(model_config.model, export=True, compile=False, load_in_8bit=False, trust_remote_code=True) # need stateful because it also enables SDPA
    if not hasattr(pt_model, 'ov_node_factory'):
        from openvino.runtime.utils.node_factory import NodeFactory
        # Keep factory to destroy it in a particular moment when all other objects referencing custom nodes are destoyed
        pt_model.ov_node_factory = NodeFactory()
        pt_model.ov_node_factory.add_extension('libuser_ov_extensions.so')
    patch_stateful_model(pt_model.model, pt_model.ov_node_factory)
    #ov.serialize(self.model.model, 'vllm_openvino_model.xml')
    core = ov.Core()
    ov_compiled = core.compile_model(pt_model.model, "CPU")
    pt_model.ov_request = ov_compiled.create_infer_request()

    from functools import partial
    pt_model._openvino_patch_orig_forward = pt_model.forward
    pt_model.forward = partial(ov_wrapper, pt_model)

    # self.vllm_model = get_model(self.model_config)
    # def sample_wrapper(*args, **kwargs):
    #     return self.vllm_model.sample(*args, hidden_states=None, **kwargs)
    # self.model.sample = sample_wrapper
    from vllm.model_executor.layers.sampler import Sampler
    pt_model.sampler = Sampler(model_config.hf_config.vocab_size)
    pt_model.sample = partial(ov_sample, pt_model)

    return pt_model
