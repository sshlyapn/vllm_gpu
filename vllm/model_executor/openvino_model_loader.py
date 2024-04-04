"""Utilities for selecting and loading models."""
from functools import partial
from typing import Optional
from pathlib import Path
import os
import torch
import numpy as np
from huggingface_hub import HfApi

from vllm.config import DeviceConfig, ModelConfig
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.input_metadata import InputMetadata
from vllm.sequence import SamplerOutput
from vllm.utils import is_openvino_optimum_intel

import openvino as ov
from openvino import Type


def _flattenize_inputs(inputs):
    """
    Helper function for making nested inputs flattens
    """
    flatten_inputs = []
    for input_data in inputs:
        if input_data is None:
            continue
        if isinstance(input_data, (list, tuple)):
            flatten_inputs.extend(_flattenize_inputs(input_data))
        elif isinstance(input_data, dict):
            flatten_inputs.extend(_flattenize_inputs(list(input_data.values())))
        else:
            flatten_inputs.append(input_data)
    return flatten_inputs

def ov_wrapper(self, *args, **kwargs) -> torch.Tensor:
    input_metadata = kwargs['input_metadata']
    flatten_kv_cache = _flattenize_inputs(kwargs['kv_caches'])

    inputs = [
        kwargs['input_ids'],
        kwargs['positions'],
        *flatten_kv_cache,
        input_metadata.is_prompt,
        input_metadata.slot_mapping
    ]

    if input_metadata.max_context_len is not None:
        # available from the second iteration
        inputs.append(input_metadata.max_context_len)
        inputs.append(input_metadata.context_lens)
        inputs.append(input_metadata.block_tables)
    else:
        inputs.append(np.array(kwargs['input_ids'].shape[1], dtype=np.int64))   # for optimum-based models this parameter can be used even on the first iteration

    self._ov_request.start_async(inputs, share_inputs=True)
    self._ov_request.wait()
    return torch.from_numpy(self._ov_request.get_tensor("logits").data)

def arguments_as_outputs(arguments):
    outputs = []
    for argument in arguments:
        if issubclass(type(argument), ov.runtime.Output):
            outputs.append(argument)
        else:
            outputs.extend(argument.outputs())
    return outputs

def patch_stateful_model(
    model: ov.Model,
    kv_cache_dtype: Type,
    is_cpu: bool):
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
    sliding_window = opset13.constant(np.array(0, np.int32))  # sliding_window

    current_seq_len = opset13.gather(opset13.shape_of(model.input('input_ids')), opset13.constant(1), opset13.constant(0))
    current_seq_len.set_friendly_name('my_current_seq_len')
    prev_max_seq_len = max_context_len - current_seq_len

    def has_parameter(model, name):
        return name in sum([list(t.get_names()) for t in model.inputs], [])

    kv_parameters = []
    assignes_to_remove = []  # not really used
    parameters_to_remove = []
    results_to_remove = []  # used, but cannot really track all Results in stateless model
    if not has_parameter(model, 'position_ids'):
        position_ids = opset13.parameter(shape=[-1, -1], dtype=np.int64, name="position_ids")
        position_ids.get_output_tensor(0).set_names({position_ids.get_friendly_name()})
        model.add_parameters([position_ids])
        print('CREATED A NEW position_ids PARAMETER')
    position_ids = model.input('position_ids')

    kv_transpose_order = opset13.constant([0, 2, 1, 3])

    class StateManagementPattern(MatcherPass):
        def __init__(self):
            MatcherPass.__init__(self)
            self.model_changed = False

            k_past_var = WrapType("opset13.ReadValue", AnyInput())
            k_past_par = WrapType("opset13.Parameter")
            k_past = Or([WrapType("opset13.Gather", [k_past_var, AnyInput(), AnyInput()]), k_past_par])
            k_past = Or([k_past, WrapType("opset13.Transpose", [k_past, AnyInput()])])  # Transpose is used when kv-cache is stored in a not usual layout, example: bloom
            k_current = AnyInput()
            k_current2 = AnyInput()
            k_current_reshaped = WrapType("opset13.Reshape", [k_current2, AnyInput()])
            k_concat = WrapType("opset13.Concat", [k_past, Or([k_current_reshaped, k_current])])

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
            v_past = Or([v_past, WrapType("opset13.Transpose", [v_past, AnyInput()])])
            v_current = AnyInput()
            v_current2 = AnyInput()
            v_current_reshaped = WrapType("opset13.Reshape", [v_current2, AnyInput()])
            v_concat = WrapType("opset13.Concat", [v_past, Or([v_current_reshaped, v_current])])

            k_shaped = kv_shaping(k_concat)
            v_shaped = kv_shaping(v_concat)

            k_simply_shaped = WrapType("opset13.Reshape", [k_concat, AnyInput()])
            v_simply_shaped = WrapType("opset13.Reshape", [v_concat, AnyInput()])

            k_order = AnyInput()
            v_order = AnyInput()

            # KV-path may already have Transposes that will be rewritten based on PA KV inputs required layout
            k_shaped_transposed = WrapType("opset13.Transpose", [Or([k_concat, k_shaped]), k_order])
            v_shaped_transposed = WrapType("opset13.Transpose", [Or([v_concat, v_shaped]), v_order])

            # Optional pattern to capture alibi slopes (based on pattern from bloom)
            alibi = AnyInput()
            sdpa_mask = WrapType("opset13.Multiply", [AnyInput(), alibi])  # apply input position_ids
            sdpa_mask = WrapType("opset13.Reshape", [sdpa_mask, AnyInput()])
            sdpa_mask = WrapType("opset13.Reshape", [sdpa_mask, AnyInput()])
            sdpa_mask = WrapType("opset13.Select", [AnyInput(), AnyInput(), sdpa_mask])

            q = AnyInput()
            sdpa = WrapType("opset13.ScaledDotProductAttention", [
                q,
                Or([k_concat, k_shaped, k_shaped_transposed, k_simply_shaped]),
                Or([v_concat, v_shaped, v_shaped_transposed, v_simply_shaped]),
                Or([sdpa_mask, AnyInput()])
            ])

            def callback(m: Matcher) -> bool:
                assert sdpa in m.get_pattern_value_map()
                mapping = m.get_pattern_value_map()
                assert sdpa in mapping
                real_q = mapping[q]

                # takes option that has 4D instead of fine-grained Reshape analysis
                # it avoids complication in the pattern, but we don't really have many options
                def take_4d(option1, option2, option3):
                    if option1 in mapping and mapping[option1].get_partial_shape().rank.get_length() == 4:
                        return mapping[option1]
                    elif mapping[option2].get_partial_shape().rank.get_length() == 4:
                        return mapping[option2]
                    else:
                        return mapping[option3]

                real_k = take_4d(k_current, k_current_reshaped, k_current2)
                real_v = take_4d(v_current, v_current_reshaped, v_current2)
                if is_cpu:
                    k_parameter = opset13.parameter(shape=[-1, -1, -1, -1], dtype=kv_cache_dtype)
                else:
                    k_parameter = opset13.parameter(shape=[-1, -1, -1, -1, -1], dtype=kv_cache_dtype)
                v_parameter = opset13.parameter(shape=[-1, -1, -1, -1], dtype=kv_cache_dtype)
                kv_parameters.append(k_parameter)
                kv_parameters.append(v_parameter)
                q_transpose = opset13.transpose(real_q, kv_transpose_order)
                q_reshape = opset13.reshape(q_transpose, opset13.constant([0, 0, -1]), True)

                k_tranpose_order = kv_transpose_order
                if k_order in mapping:  # reapply transpose found in the graph by manipulating of indices of our Transpose
                    k_tranpose_order = opset13.gather(mapping[k_order], kv_transpose_order, opset13.constant(0))
                k_transpose = opset13.transpose(real_k, k_tranpose_order)
                k_reshape = opset13.reshape(k_transpose, opset13.constant([0, 0, -1]), True)

                v_tranpose_order = kv_transpose_order
                if v_order in mapping:  # reapply transpose found in the graph by manipulating of indices of our Transpose
                    v_tranpose_order = opset13.gather(mapping[v_order], kv_transpose_order, opset13.constant(0))
                v_transpose = opset13.transpose(real_v, v_tranpose_order)
                v_reshape = opset13.reshape(v_transpose, opset13.constant([0, 0, -1]), True)

                # TODO: Detect whether SDPA in the model graph has `scale` argument set and use it instead of the computed scale below
                # Most likely `scale` will always be a constant in real inference, but dynamic dimension propagation may not always derive it as a constant
                # That's why a sub-graph computing `scale` is built instead of just a constant node.
                hidden_shape = opset13.shape_of(real_q)
                hidden_dim = opset13.gather(hidden_shape, opset13.constant(-1), opset13.constant(0))
                scale = opset13.constant(1.0, dtype=ov.Type.f32)/opset13.sqrt(opset13.convert(hidden_dim, destination_type=ov.Type.f32))

                if alibi in mapping:
                    print('alibi slopes applied')
                    alibi_slopes = opset13.reshape(mapping[alibi], opset13.constant([-1]), special_zero=False)
                    if alibi_slopes.get_element_type() != ov.Type.f32:
                        alibi_slopes = opset13.convert(alibi_slopes, destination_type=ov.Type.f32)  #todo
                else:
                    alibi_slopes = opset13.constant(np.array([], np.float32))

                paged_attention = ov.runtime.op._PagedAttentionExtension(arguments_as_outputs([
                    q_reshape,
                    k_reshape,
                    v_reshape,
                    k_parameter,
                    v_parameter,
                    *model_remaining_params,
                    scale,
                    alibi_slopes,
                    sliding_window
                ]))
                pa_shape = opset13.concat([
                        opset13.constant([0]),
                        opset13.constant([0]),
                        opset13.constant([-1]),
                        opset13.unsqueeze(hidden_dim, opset13.constant(0))
                    ], axis=0)
                pa_reshape = opset13.reshape(paged_attention, pa_shape, True)
                pa_transpose = opset13.transpose(pa_reshape, kv_transpose_order)

                #TODO: Complete this part to work with stateless models as well as will stateful
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

            self.register_matcher(Matcher(sdpa, "StateManagementPattern"), callback)

    class PrevSequenceLengthPattern(MatcherPass):
        def __init__(self):
            MatcherPass.__init__(self)
            self.model_changed = False

            kv_past = WrapType("opset13.ReadValue", AnyInput())
            kv_gather = WrapType("opset13.Gather", [kv_past, AnyInput(), AnyInput()])
            kv_shape = WrapType("opset13.ShapeOf", [kv_gather])
            seq = WrapType("opset13.Gather", [kv_shape, AnyInput(), AnyInput()])

            def callback(m: Matcher) -> bool:
                # TODO: Check that seq has axis that really takes sequence len but not any other dimension -- use symbolics or look at the constant input
                gather = m.get_match_root()
                target_type = gather.get_output_element_type(0)
                if prev_max_seq_len.get_output_element_type(0) != target_type:
                    print(f'Converting {prev_max_seq_len.get_output_element_type(0)} of max_context_len to {target_type}')
                    replacement = opset13.convert(prev_max_seq_len, target_type)
                else:
                    replacement = prev_max_seq_len
                replace_node(gather, replacement)
                print("DETECTED PATTERN PrevSequenceLengthPattern, CONNECTED TO A DEDICATED PARAMETER")
                return True

            self.register_matcher(Matcher(seq, "PrevSequenceLengthPattern"), callback)

    class TotalSequenceLengthPattern(MatcherPass):
        def __init__(self):
            MatcherPass.__init__(self)
            self.model_changed = False

            kv_past = WrapType("opset13.ReadValue", AnyInput())
            kv_gather = WrapType("opset13.Gather", [kv_past, AnyInput(), AnyInput()])
            kv_current = AnyInput()
            kv_concat = WrapType("opset13.Concat", [kv_gather, kv_current])
            kv_shape = WrapType("opset13.ShapeOf", [kv_concat])
            seq = WrapType("opset13.Gather", [kv_shape, AnyInput(), AnyInput()])

            def callback(m: Matcher) -> bool:
                # TODO: Check that seq has axis that really takes sequence len but not any other dimension -- use symbolic infra or look at the constant input
                gather = m.get_match_root()
                target_type = gather.get_output_element_type(0)
                if max_context_len.get_output_element_type(0) != target_type:
                    print(f'Converting {max_context_len.get_output_element_type(0)} of total_seq_len to {target_type}')
                    replacement = opset13.convert(max_context_len, target_type)
                else:
                    replacement = max_context_len
                replace_node(gather, replacement)
                print("DETECTED PATTERN TotalSequenceLengthPattern, CONNECTED TO A DEDICATED PARAMETER")
                return True

            self.register_matcher(Matcher(seq, "TotalSequenceLengthPattern"), callback)

    # TODO: Instead of using the following transformation that matches quite a specific place in a model graph in case when position_ids parameter is missing,
    #       consider replacing always existing attention_mask parameter with a sub-graph using a new slot_mapping parameter.
    class PositionIDsReplacer(MatcherPass):
        def __init__(self):
            MatcherPass.__init__(self)
            self.model_changed = False

            input_ids = AnyInput()
            input_embed = WrapType("opset13.Gather", [AnyInput(), input_ids, AnyInput()])

            position_ids_pattern = AnyInput()
            offset = WrapType('opset13.Constant')
            add_offset = WrapType('opset13.Add', [position_ids_pattern, offset])
            convert = WrapType('opset13.Convert', [add_offset])
            position_embed = WrapType("opset13.Gather", [AnyInput(), convert, AnyInput()])

            add = WrapType("opset13.Add", [input_embed, position_embed])

            def callback(m: Matcher) -> bool:
                mapping = m.get_pattern_value_map()
                replace_node(mapping[position_ids_pattern].get_node(), position_ids.get_node())
                print('APPLIED position_ids PARAMETER INSTEAD OF attention_mask-BASED SUB-GRAPH')
                return True

            self.register_matcher(Matcher(add, "PositionIDsReplacer"), callback)

    m = Manager()
    m.set_per_pass_validation(False)
    m.register_pass(StateManagementPattern())
    m.register_pass(PrevSequenceLengthPattern())
    m.register_pass(TotalSequenceLengthPattern())

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
    # Remove all Assigns aggressively, the path from the kv-cache concat to Assign can be complicated,
    # but there is no reason to track it and reject part of the Assigns, because the model will remain
    # in incorrect form anyway.
    sinks = model.get_sinks()
    for sink in sinks:
        model.remove_sink(sink)
    for result in results_to_remove:
        model.remove_result(result)
    model.add_parameters(kv_parameters)
    model.add_parameters(model_remaining_params)
    print('PARAMETERS ARE REORGANIZED, THE STATE (IF EXISTS) IS REMOVED')

def _patch_model_with_openvino(
        pt_model: torch.nn.Module,
        model_config: ModelConfig,
        kv_cache_dtype: Type,
        is_cpu: bool):
    print(' ============= PATCHING MODEL =============')
    from vllm.model_executor.layers.attention.attention import Attention
    from openvino.frontend.pytorch import ModuleExtension, ConversionExtension
    from openvino import Core, convert_model, Type, PartialShape

    # Avoid usage of vllm._C.ops

    from vllm.model_executor.layers.activation import SiluAndMul, NewGELU, FastGELU
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding

    SiluAndMul.forward = SiluAndMul._forward
    NewGELU.forward = NewGELU._forward
    FastGELU.forward = FastGELU._forward
    RMSNorm.forward = RMSNorm._forward
    RotaryEmbedding.forward = RotaryEmbedding._forward

    # Prepare example inputs

    torch_dtype_maping = {
        Type.boolean: torch.bool,
        Type.f32: torch.float32,
        Type.f16: torch.float16,
        Type.bf16: torch.bfloat16,
        Type.i32: torch.int32,
        Type.i64: torch.int64
    }
    kv_cache_dtype = torch_dtype_maping[kv_cache_dtype]
    num_heads = pt_model.config.num_attention_heads
    num_kv_heads = num_heads
    head_size = pt_model.config.hidden_size // num_kv_heads
    num_hidden_layers = model_config.hf_config.num_hidden_layers

    _PAD_SLOT_ID = -1
    _EXAMPLE_BLOCK_SIZE = 8
    _EXAMPLE_NUM_BLOCKS = 256
    _X = kv_cache_dtype.itemsize

    _BATCH_SIZES_TO_CAPTURE = [2]
    max_batch_size = max(_BATCH_SIZES_TO_CAPTURE)
    max_context_len = (
                model_config.max_context_len_to_capture
                if model_config is not None else 0)
    max_num_blocks = (max_context_len + _EXAMPLE_BLOCK_SIZE - 1) // _EXAMPLE_BLOCK_SIZE

    slot_mapping = torch.empty(max_batch_size, 1, dtype=torch.long)
    slot_mapping.fill_(_PAD_SLOT_ID)
    context_lens = torch.ones(max_batch_size, dtype=torch.int32)
    prompt_lens = torch.ones(max_batch_size, dtype=torch.long)
    start_loc = torch.ones(max_batch_size, dtype=torch.long)
    block_tables = torch.ones((max_batch_size, max_num_blocks), dtype=torch.int32)

    if is_cpu:
        kv_cache = [(torch.ones((_EXAMPLE_NUM_BLOCKS, num_kv_heads, _EXAMPLE_BLOCK_SIZE, head_size), dtype=kv_cache_dtype),
                    torch.ones((_EXAMPLE_NUM_BLOCKS, num_kv_heads, _EXAMPLE_BLOCK_SIZE, head_size), dtype=kv_cache_dtype))] * num_hidden_layers
    else:
        kv_cache = [(torch.ones((_EXAMPLE_NUM_BLOCKS, num_kv_heads, head_size // _X, _EXAMPLE_BLOCK_SIZE, _X), dtype=kv_cache_dtype),
                    torch.ones((_EXAMPLE_NUM_BLOCKS, num_kv_heads, head_size, _EXAMPLE_BLOCK_SIZE), dtype=kv_cache_dtype))] * num_hidden_layers

    input_meta = {
        "is_prompt": torch.tensor(False),
        "slot_mapping": slot_mapping,
        "max_seq_len": torch.tensor(256),
        "max_context_len": torch.tensor(max_context_len),
        "context_lens": context_lens,
        "block_tables": block_tables,
        "prompt_lens": prompt_lens,
        "start_loc": start_loc,
        "use_cuda_graph": torch.tensor(False),
        "kv_cache_dtype": torch.tensor(False), # TODO: openvino.tools.ovc.error.Error: Unexpected type of example_input. Supported types torch.Tensor, np.array or ov.Tensor. Got <class 'str'>
    }

    example_input = (torch.ones((1, 1), dtype=torch.long), torch.range(0, 10, dtype=torch.long).unsqueeze(0)[:, -1:], tuple(kv_cache), input_meta)

    class ModelWrapper(torch.nn.Module):
        '''
        Model wrapper to convert a map of aatributes to InputMetadata struct
        '''

        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, position_ids, kv_cache, meta_dict):
            input_meta = InputMetadata(**meta_dict)
            return self.model(input_ids, position_ids, kv_cache, input_meta)

    def wrapper(module, target_op, *args, **kwargs):
        # this function will replace entier PageAttention module
        # target_op is PageAttentionExtension, the order of arguments below should match the extension signature
        return target_op(
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            args[5].is_prompt,
            args[5].slot_mapping,
            args[5].max_context_len,
            args[5].context_lens,
            args[5].block_tables,
            torch.tensor(module.backend.scale),  # wrap in a tensor, otherwise it will not appear in the trace
            torch.tensor(module.backend.alibi_slopes if module.backend.alibi_slopes is not None else [], dtype=torch.float32),  # alibi_slopes
            torch.tensor(module.backend.sliding_window if module.backend.sliding_window is not None else 0, dtype=torch.int32)  # sliding_window
        )

    def paged_attention_convertion(context):
        inputs = [context.get_input(i) for i in range(context.get_input_size())]
        pa = ov.runtime.op._PagedAttentionExtension(inputs)
        return pa.outputs()

    with torch.no_grad():
        print('>>>>>>>>>>>>> CONVERTING OV MODEL')
        ov_model =  convert_model(
            ModelWrapper(pt_model),
            example_input=example_input,
            extension=[
                ModuleExtension(
                    Attention,
                    target_op='PagedAttentionExtension',
                    evaluate=lambda module, *args, **kwargs: args[0],  # need this because PagedAttention module fails in torch.jit.trace
                    convert=wrapper
                ),
                ConversionExtension('PagedAttentionExtension', paged_attention_convertion),
            ]
        )

        ov_dtype_maping = {
            torch.bool: Type.boolean,
            torch.float32: Type.f32,
            torch.float16: Type.f16,
            torch.bfloat16: Type.bf16,
            torch.int32: Type.i32,
            torch.int64: Type.i64
        }

        for example_input_data, input_tensor in zip(_flattenize_inputs(example_input), ov_model.inputs):
            if input_tensor.element_type.is_dynamic():
                input_tensor.get_node().set_element_type(ov_dtype_maping[example_input_data.dtype])
            if input_tensor.partial_shape.rank.is_dynamic:
                input_tensor.get_node().set_partial_shape(PartialShape([-1]*example_input_data.ndim))

        for out_name, out in zip(["logits"], ov_model.outputs):
            out.get_tensor().set_names({out_name})
        ov_model.validate_nodes_and_infer_types()
        print('>>>>>>>>>>>>> OV MODEL CONVERTED')

    core = Core()
    ov_compiled_model = core.compile_model(ov_model, "CPU")
    ov_request = ov_compiled_model.create_infer_request()

    pt_model._ov_request = ov_request
    pt_model._openvino_patch_orig_forward = pt_model.forward
    pt_model.forward = partial(ov_wrapper, pt_model)


def ov_sample(
    self,
    hidden_states: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> Optional[SamplerOutput]:
    return self.sampler(None, hidden_states, sampling_metadata)


def require_model_export(model_id, revision=None, subfolder=None):
    # Stored IR may not be suitable for vLLM purposes (too old, not stateful, not compressed etc.)
    # This is an option to override IR usage logic and alway do model conversion.
    if os.environ.get('VLLM_OPENVINO_OPTIMUM_FORCE_CONVERSION', '0') == '1':
        return True
    model_dir = Path(model_id)
    if subfolder is not None:
        model_dir = model_dir / subfolder
    if model_dir.is_dir():
        return not (model_dir / "openvino_model.xml").exists() or not (model_dir / "openvino_model.bin").exists()

    hf_api =  HfApi()
    try:
        model_info = hf_api.model_info(model_id, revision=revision or "main")
        normalized_subfolder = None if subfolder is None else Path(subfolder).as_posix()
        model_files = [file.rfilename for file in model_info.siblings if normalized_subfolder is None or file.rfilename.startswith(normalized_subfolder)]
        ov_model_path = "openvino_model.xml" if normalized_subfolder is None else f"{normalized_subfolder}/openvino_model.xml"
        return not ov_model_path in model_files or not ov_model_path.replace(".xml", ".bin") in model_files
    except Exception:
         return True


def get_model(model_config: ModelConfig,
              device_config: DeviceConfig,
              kv_cache_dtype: Type,
              **kwargs) -> torch.nn.Module:
    lora_config = kwargs.get("lora_config", None)
    if lora_config:
        raise ValueError(
            f"OpenVINO modeling does not support LoRA, "
            "but LoRA is enabled. Support for this model may "
            "be added in the future. If this is important to you, "
            "please open an issue on github.")

    pt_model = None

    if is_openvino_optimum_intel():
        import openvino as ov
        from optimum.intel import OVModelForCausalLM
        export = require_model_export(model_config.model)
        if export:
            print(f'[ INFO ] Provided model id {model_config.model} does not contain OpenVINO IR, the model will be converted to IR with default options. '
                  'If you need to use specific options for model conversion, use optimum-cli export openvino with desired options.')
        else:
            print(f'[ INFO ] OpenVINO IR is avaialble for provided model id {model_config.model}. '
                  'This IR will be used for inference as-is, all possible options that may affect model conversion are ignored.')
        pt_model = OVModelForCausalLM.from_pretrained(
            model_config.model,
            export=export,
            compile=False,
            load_in_8bit=False,
            trust_remote_code=model_config.trust_remote_code
        )
        patch_stateful_model(pt_model.model, kv_cache_dtype, device_config.device.type == "cpu")

        # For deployment outside vLLM
        model_file_name = os.environ.get('VLLM_OPENVINO_EXPORTED_IR_NAME', '')
        if model_file_name:
            ov.save_model(pt_model.model, model_file_name)

        core = ov.Core()
        ov_compiled = core.compile_model(pt_model.model, "CPU")
        pt_model._ov_request = ov_compiled.create_infer_request()

        pt_model._openvino_patch_orig_forward = pt_model.forward
        pt_model.forward = partial(ov_wrapper, pt_model)

        from vllm.model_executor.layers.sampler import Sampler
        pt_model.sampler = Sampler(model_config.hf_config.vocab_size)
        pt_model.sample = partial(ov_sample, pt_model)
    else:
        from vllm.model_executor.model_loader import get_model
        pt_model = get_model(model_config, device_config, **kwargs)
        _patch_model_with_openvino(pt_model, model_config, kv_cache_dtype, device_config.device.type == "cpu")

    return pt_model
