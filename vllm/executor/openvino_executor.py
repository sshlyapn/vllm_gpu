from typing import Dict, List, Optional, Tuple, Set
import torch.distributed

from vllm.lora.request import LoRARequest
from vllm.config import (CacheConfig, DeviceConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig, LoRAConfig)
from vllm.executor.executor_base import ExecutorAsyncBase, ExecutorBase
from vllm.executor.utils import check_block_size_valid
from vllm.logger import init_logger
from vllm.sequence import SamplerOutput, SequenceGroupMetadata
from vllm.worker.model_runner import ModelRunner
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.model_executor.parallel_utils.parallel_state import ensure_model_parallel_initialized
from vllm.utils import (get_ip, get_open_port, get_distributed_init_method,
                        make_async, STR_DTYPE_TO_TORCH_DTYPE)

logger = init_logger(__name__)


KVCache = Tuple[torch.Tensor, torch.Tensor]

class OpenVINOCacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config

        self.head_size = model_config.get_head_size()
        self.num_layers = model_config.get_num_layers(parallel_config)
        self.num_heads = model_config.get_num_kv_heads(parallel_config)

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        self.num_cpu_blocks = cache_config.num_cpu_blocks

        if cache_config.cache_dtype == "auto":
            self.cache_dtype = model_config.dtype
        else:
            self.cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # Initialize the cache.
        self.gpu_cache = self.allocate_gpu_cache()
        self.cpu_cache = self.allocate_cpu_cache()

    def get_key_block_shape(self) -> Tuple[int, int, int, int]:
        element_size = torch.tensor([], dtype=self.cache_dtype).element_size()
        x = 16 // element_size
        return (
            self.num_heads,
            self.head_size // x,
            self.block_size,
            x,
        )

    def get_value_block_shape(self) -> Tuple[int, int, int]:
        return (
            self.num_heads,
            self.head_size,
            self.block_size,
        )

    def allocate_gpu_cache(self) -> List[KVCache]:
        # TODO: use OpenVINO RemoteTensor API
        if self.num_gpu_blocks == 0:
            return None

        gpu_cache: List[KVCache] = []
        return gpu_cache

    def allocate_cpu_cache(self) -> List[KVCache]:
        cpu_cache: List[KVCache] = []
        key_block_shape = self.get_key_block_shape()
        value_block_shape = self.get_value_block_shape()
        for _ in range(self.num_layers):
            key_blocks = torch.empty(
                size=(self.num_cpu_blocks, *key_block_shape),
                dtype=self.cache_dtype,
                device="cpu",
            )
            value_blocks = torch.empty(
                size=(self.num_cpu_blocks, *value_block_shape),
                dtype=self.cache_dtype,
                device="cpu",
            )
            cpu_cache.append((key_blocks, value_blocks))
        return cpu_cache

    def _swap(
        self,
        src: List[KVCache],
        dst: List[KVCache],
        src_to_dst: Dict[int, int],
    ) -> None:
        # TODO: ilavreno: implement cache sync via OpenVINO RemoteContext API host <-> device
        assert False

    def swap_in(self, src_to_dst: Dict[int, int]) -> None:
        self._swap(self.cpu_cache, self.gpu_cache, src_to_dst)

    def swap_out(self, src_to_dst: Dict[int, int]) -> None:
        self._swap(self.gpu_cache, self.cpu_cache, src_to_dst)

    def copy(self, src_to_dsts: Dict[int, List[int]]) -> None:
        key_caches = [key_cache for key_cache, _ in self.gpu_cache]
        value_caches = [value_cache for _, value_cache in self.gpu_cache]
        # TODO: ilavreno: implement cache sync via OpenVINO RemoteContext API device -> device
        # cache_ops.copy_blocks(key_caches, value_caches, src_to_dsts)

    @staticmethod
    def get_cache_block_size(
        block_size: int,
        cache_dtype: str,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_layers = model_config.get_num_layers(parallel_config)

        key_cache_block = block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_layers * (key_cache_block + value_cache_block)
        if cache_dtype == "auto":
            cache_dtype = model_config.dtype
        else:
            cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
        cache_dtype_size = cache_dtype.itemsize
        return cache_dtype_size * total


class OpenVINOWorker:
    """A worker class that executes the model on OpenVINO device.

    Currently, OpenVINO supports a single worker at the moment. The worker is
    responsible for maintaining the KV cache and executing the model on
    OpenVINO device.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        lora_config: Optional[LoRAConfig] = None,
        kv_cache_dtype: Optional[str] = "auto",
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.lora_config = lora_config

        # Load FP32 models to OpenVINO
        model_config.dtype = torch.float32

        self.model_runner = ModelRunner(model_config,
                                        parallel_config,
                                        scheduler_config,
                                        device_config,
                                        lora_config=self.lora_config,
                                        kv_cache_dtype=kv_cache_dtype,
                                        is_driver_worker=True)
        # Uninitialized cache engine. Will be initialized by self.init_cache_engine().
        self.cache_config = None
        self.cache_engine = None
        self.gpu_cache = None
        self.cpu_cache = None

        # Not required for OpenVINO, but without torch.distributed initialization all CC operations failed
        self._init_distributed_environment()

    def init_model(self) -> None:
        # Initialize the model.
        set_random_seed(self.model_config.seed)

    def load_model(self):
        self.model_runner.load_model()

    def profile_num_available_blocks(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        cpu_swap_space: int,
        cache_dtype: str,
    ) -> Tuple[int, int]:
        """Profiles the peak memory usage of the model and returns the maximum
        number of Accelerator and CPU cache blocks that can be allocated.

        Args:
            block_size: The size of the cache block.
            gpu_memory_utilization: The fraction of the total GPU memory to use.
            cpu_swap_space: The size of the CPU swap space in bytes.
        """
        if self.device_config.device.type == 'cpu':
            cache_block_size = OpenVINOCacheEngine.get_cache_block_size(
                block_size, cache_dtype, self.model_config, self.parallel_config)
            num_gpu_blocks = 0
            num_cpu_blocks = int(cpu_swap_space // cache_block_size)
            num_cpu_blocks = max(num_cpu_blocks, 0)

            return num_gpu_blocks, num_cpu_blocks

        # TODO: ilavreno: profile GPU run and compute available memory which can be
        # used for KV cache blocks

    def init_cache_engine(self, cache_config: CacheConfig) -> None:
        self.cache_config = cache_config
        self.cache_engine = OpenVINOCacheEngine(self.cache_config, self.model_config,
                                                self.parallel_config)
        self.gpu_cache = self.cache_engine.gpu_cache
        self.cpu_cache = self.cache_engine.cpu_cache
        self.model_runner.set_block_size(self.cache_engine.block_size)

    def cache_swap(
        self,
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> None:
        # Issue cache operations.
        issued_cache_op = False
        if blocks_to_swap_in:
            self.cache_engine.swap_in(blocks_to_swap_in)
            issued_cache_op = True
        if blocks_to_swap_out:
            self.cache_engine.swap_out(blocks_to_swap_out)
            issued_cache_op = True
        if blocks_to_copy:
            self.cache_engine.copy(blocks_to_copy)
            issued_cache_op = True

        cache_events = self.cache_events if issued_cache_op else None

        # Wait for cache operations to finish.
        # TODO: ilavrenov, probably, we need similar for GPU
        if cache_events is not None:
            for event in cache_events:
                event.wait()

    def execute_model(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]] = None,
        blocks_to_swap_in: Optional[Dict[int, int]] = None,
        blocks_to_swap_out: Optional[Dict[int, int]] = None,
        blocks_to_copy: Optional[Dict[int, List[int]]] = None,
    ) -> Optional[SamplerOutput]:
        self.cache_swap(blocks_to_swap_in, blocks_to_swap_out, blocks_to_copy)

        # If there is no input, we don't need to execute the model.
        if len(seq_group_metadata_list) == 0:
            return {}

        # TODO: ilavreno: currently device.type cannot be GPU, need to find a way to pass device to OpenVINO backend
        device_cache = self.gpu_cache if self.device_config.device.type == 'gpu' else self.cpu_cache
        output = self.model_runner.execute_model(seq_group_metadata_list, device_cache)
        return output

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def list_loras(self) -> Set[int]:
        return self.model_runner.list_loras()

    @property
    def max_model_len(self) -> int:
        return self.model_config.max_model_len

    @property
    def vocab_size(self) -> int:
        return self.model_runner.vocab_size

    def get_cache_block_size_bytes(self, block_size: int,
                                   cache_dtype: str) -> int:
        """Get the size of the KV cache block size in bytes.
        """
        return OpenVINOCacheEngine.get_cache_block_size(block_size, cache_dtype,
                                                        self.model_config,
                                                        self.parallel_config)

    def _init_distributed_environment(self) -> None:
        """Initialize the distributed environment."""
        distributed_init_method = get_distributed_init_method(
            get_ip(), get_open_port())
        rank = 0
        backend = "gloo"

        if torch.distributed.is_initialized():
            torch_world_size = torch.distributed.get_world_size()
            if torch_world_size != self.parallel_config.world_size:
                raise RuntimeError(
                    "torch.distributed is already initialized but the torch world "
                    "size does not match parallel_config.world_size "
                    f"({torch_world_size} vs. {self.parallel_config.world_size}).")
        elif not distributed_init_method:
            raise ValueError(
                "distributed_init_method must be set if torch.distributed "
                "is not already initialized")
        else:
            torch.distributed.init_process_group(
                backend=backend,
                world_size=self.parallel_config.world_size,
                rank=rank,
                init_method=distributed_init_method,
            )

        # A small all_reduce for warmup.
        torch.distributed.all_reduce(torch.zeros(1, device=self.device_config.device))
        ensure_model_parallel_initialized(self.parallel_config.tensor_parallel_size,
                                          self.parallel_config.pipeline_parallel_size)

def _patch_model_with_openvino(model, model_config):
    if hasattr(model, '_openvino_patch_orig_forward'):
        return
    print(' ============= PATCHING MODEL =============')
    # model._openvino_patch_orig_forward = model.forward
    # Replace forward with our stuff
    import openvino as ov

    import torch
    import numpy as np
    import openvino as ov
    from vllm.model_executor.layers.attention.attention import Attention
    from openvino.frontend.pytorch import ModuleExtension

    from typing import Optional

    import torch
    from dataclasses import dataclass

    @dataclass
    class InputMetadata:
        """Metadata for input sequences. Used in PagedAttention.

        Args:
            prompt_lens: Lengths of prompts.
            slot_mapping: The address to write the new KV to of each token.
            max_context_len: The maximum context length.
            context_lens: the length of attention context for each sequence.
            block_tables: The block tables. (Seq id -> list of physical block)
            kv_cache_dtype: Data type to store kv cache.
        """

        def __init__(
            self,
            is_prompt: bool = False,
            slot_mapping: torch.Tensor = None,
            prompt_lens: Optional[torch.Tensor] = None,
            max_seq_len: Optional[int] = None,
            start_loc: Optional[torch.Tensor] = None,
            max_context_len: Optional[int] = None,
            context_lens: Optional[torch.Tensor] = None,
            block_tables: Optional[torch.Tensor] = None,
            use_cuda_graph: bool = False,
            kv_cache_dtype: str = "auto",
        ) -> None:
            self.is_prompt = is_prompt
            self.prompt_lens = prompt_lens
            self.max_seq_len = max_seq_len
            self.start_loc = start_loc
            self.max_context_len = max_context_len
            self.slot_mapping = slot_mapping
            self.context_lens = context_lens
            self.block_tables = block_tables
            self.use_cuda_graph = use_cuda_graph
            self.kv_cache_dtype = kv_cache_dtype

            # Set during the execution of the first attention op.
            # FIXME(woosuk): This is a hack.
            self.attn_bias = None

        def __repr__(self) -> str:
            return ("InputMetadata("
                    f"is_prompt={self.is_prompt}, "
                    f"max_context_len={self.max_context_len}, "
                    f"slot_mapping={self.slot_mapping}, "
                    f"context_lens={self.context_lens}, "
                    f"block_tables={self.block_tables}, "
                    f"use_cuda_graph={self.use_cuda_graph}, "
                    f"kv_cache_dtype={self.kv_cache_dtype})")

    _PAD_SLOT_ID = -1

    pt_model = model
    _BATCH_SIZES_TO_CAPTURE = [2]
    max_batch_size = max(_BATCH_SIZES_TO_CAPTURE)
    input_tokens = torch.zeros(max_batch_size, 1, dtype=torch.long)
    input_positions = torch.zeros(max_batch_size, 1,
                                        dtype=torch.long)
    slot_mapping = torch.empty(max_batch_size, 1, dtype=torch.long)
    slot_mapping.fill_(_PAD_SLOT_ID)
    context_lens = torch.ones(max_batch_size, dtype=torch.int32)

    max_context_len_to_capture = (
                model_config.max_context_len_to_capture
                if model_config is not None else 0)
    block_size = 8
    max_num_blocks = (max_context_len_to_capture + block_size -
                            1) // block_size
    graph_block_tables = np.zeros(
                (max(_BATCH_SIZES_TO_CAPTURE), max_num_blocks), dtype=np.int32)

    block_tables = torch.from_numpy(graph_block_tables)

    #TODO: Take real max_seq_len from somewhere
    input_meta = {"is_prompt": torch.tensor(False), "slot_mapping": slot_mapping, "max_seq_len": torch.tensor(256), "max_context_len": torch.tensor(2048), "context_lens": context_lens, "block_tables": block_tables}

    fp_type = torch.float32
    num_heads = pt_model.config.num_attention_heads
    head_size = pt_model.config.hidden_size
    head_dim = head_size // num_heads

    # // value_cache: shape = [num_blocks, num_kv_heads, head_size, block_size]
    # // key_cache: shape [num_blocks, num_kv_heads, head_size/x, block_size, x]
    #TODO: Take example tensors from model_args/model_kwargs
    kv_cache = [(torch.ones((3640, 12, 16, 16, 4), dtype=fp_type), torch.ones((3640, 12, 64, 16), dtype=fp_type))] * model_config.hf_config.num_hidden_layers

    example_input = (torch.ones((1, 1), dtype=torch.long), torch.range(0, 10, dtype=torch.long).unsqueeze(0)[:, -1:], tuple(kv_cache), input_meta)
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, position_ids, kv_cache, meta_dict):
            input_meta = InputMetadata(**meta_dict)
            return self.model(input_ids, position_ids, kv_cache, input_meta)


    model_wrapper = ModelWrapper(pt_model)

    ov_dtype_maping = {
        torch.bool: ov.Type.boolean,
        torch.float32: ov.Type.f32,
        torch.float16: ov.Type.f16,
        torch.bfloat16: ov.Type.bf16,
        torch.int32: ov.Type.i32,
        torch.int64: ov.Type.i64
    }

    # avoid usage of vllm._C.ops
    from vllm.model_executor.layers.activation import SiluAndMul, NewGELU, FastGELU
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding

    SiluAndMul.forward = SiluAndMul._forward
    NewGELU.forward = NewGELU._forward
    FastGELU.forward = FastGELU._forward
    RMSNorm.forward = RMSNorm._forward
    RotaryEmbedding.forward = RotaryEmbedding._forward

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

    flatten_input = flattenize_inputs(example_input)
    input_names = ["input_ids", "position_ids"]
    output_names = ["logits"]

    for i in range(12):
        input_names.extend([f"past_key_values.{i}.key", f"past_key_values.{i}.value"])

    input_names.extend(list(input_meta))

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

    with torch.no_grad():
        print('>>>>>>>>>>>>> CONVERTING OV MODEL')
        ov_model =  ov.convert_model(
            model_wrapper,
            example_input=example_input,
            extension=[
                ModuleExtension(
                    Attention,
                    target_op='PagedAttentionExtension',
                    evaluate=lambda module, *args, **kwargs: args[0],  # need this because PagedAttention module fails in torch.jit.trace
                    convert=wrapper
                ),
                "libuser_ov_extensions.so"
            ]
        )

        for input_data, input_tensor in zip(flatten_input, ov_model.inputs):
            if input_tensor.element_type.is_dynamic():
                input_tensor.get_node().set_element_type(ov_dtype_maping[input_data.dtype])
            if input_tensor.partial_shape.rank.is_dynamic:
                input_tensor.get_node().set_partial_shape(ov.PartialShape([-1]*input_data.ndim))

        for out_name, out in zip(output_names, ov_model.outputs):
            out.get_tensor().set_names({out_name})
        ov_model.validate_nodes_and_infer_types()
        # ov.save_model(ov_model, "vllm_openvino_model.xml")
        print('>>>>>>>>>>>>> OV MODEL CONVERTED')
        #print(ov_model)

    core = ov.Core()
    core.add_extension("libuser_ov_extensions.so")
    ov_config = {ov.properties.enable_profiling: True}
    # ov_config = {}
    ov_compiled = core.compile_model(ov_model, "CPU", config=ov_config)
    ov_request = ov_compiled.create_infer_request()

    from functools import partial
    def wrapper(*args, **kwargs):
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
        #for input in inputs:
        #    print(f'{input.dtype} wiht shape {input.shape}' if isinstance(input, torch.Tensor) else type(input))
        result = ov_request.infer(inputs, share_inputs=True, share_outputs=False)
        #print(f'result: {type(result)}')
        return torch.from_numpy(result[0])
    model._openvino_patch_orig_forward = model.forward
    model.forward = partial(wrapper, model)


class OpenVINOExecutor(ExecutorBase):

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        lora_config: Optional[LoRAConfig],
    ) -> None:
        self.model_config = model_config
        self.cache_config = cache_config
        self.lora_config = lora_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config

        # Instantiate the worker and load the model to OpenVINO device.
        self._init_worker()

        # Profile the memory usage and initialize the cache.
        self._init_cache()

    def _init_worker(self):
        assert self.parallel_config.world_size == 1, (
            "OpenVINO worker only supports single inference device.")

        self.driver_worker = OpenVINOWorker(
            self.model_config,
            self.parallel_config,
            self.scheduler_config,
            self.device_config,
            lora_config=self.lora_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
        )
        self.driver_worker.init_model()
        self.driver_worker.load_model()

        _patch_model_with_openvino(self.driver_worker.model_runner.model, self.model_config)

    def _init_cache(self) -> None:
        """Profiles the memory usage and initializes the KV cache.

        The engine first profiles the existing memory usage.
        Then, it allocates the remaining memory for KV blocks.

        .. tip::
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        # Get the maximum number of blocks that can be allocated on GPU and CPU.
        num_gpu_blocks, num_cpu_blocks = (
            self.driver_worker.profile_num_available_blocks(
                block_size=self.cache_config.block_size,
                gpu_memory_utilization=self.cache_config.
                gpu_memory_utilization,
                cpu_swap_space=self.cache_config.swap_space_bytes,
                cache_dtype=self.cache_config.cache_dtype,
            ))

        logger.info(f"# GPU blocks: {num_gpu_blocks}, "
                    f"# CPU blocks: {num_cpu_blocks}")

        # OpenVINO operates with CPU blocks
        check_block_size_valid(num_cpu_blocks, self.cache_config.block_size,
                               self.model_config.max_model_len)

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        # Initialize the cache.
        self.driver_worker.init_cache_engine(cache_config=self.cache_config)

    def execute_model(self,
                      seq_group_metadata_list: List[SequenceGroupMetadata],
                      blocks_to_swap_in: Dict[int, int],
                      blocks_to_swap_out: Dict[int, int],
                      blocks_to_copy: Dict[int, List[int]]) -> SamplerOutput:
        output = self.driver_worker.execute_model(
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
        )
        return output

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def list_loras(self) -> List[int]:
        return self.model_runner.list_loras()

    def check_health(self) -> None:
        # OpenVINO will always be healthy as long as it's running.
        return


class OpenVINOExecutorAsync(OpenVINOExecutor, ExecutorAsyncBase):

    async def execute_model_async(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> SamplerOutput:
        output = await make_async(self.driver_worker.execute_model)(
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy)
        return output

    async def check_health_async(self) -> None:
        # OpenVINO will always be healthy as long as it's running.
        return
