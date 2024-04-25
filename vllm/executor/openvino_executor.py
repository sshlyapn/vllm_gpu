from typing import Dict, List, Optional, Tuple, Set
import torch.distributed
import gc
import os

from vllm.lora.request import LoRARequest
from vllm.config import (CacheConfig, DeviceConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig, LoRAConfig)
from vllm.executor.executor_base import ExecutorAsyncBase, ExecutorBase
from vllm.executor.utils import check_block_size_valid
from vllm.logger import init_logger
from vllm.sequence import SamplerOutput, SequenceGroupMetadata, SequenceData
from vllm.worker.model_runner import ModelRunner
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.model_executor.parallel_utils.parallel_state import ensure_model_parallel_initialized
from vllm.utils import (get_ip, get_open_port, get_distributed_init_method,
                        make_async, STR_DTYPE_TO_TORCH_DTYPE)
from vllm.sampling_params import SamplingParams

import openvino as ov
import openvino.properties.hint as hints

logger = init_logger(__name__)

OpenVINOKVCache = Tuple[ov.Tensor, ov.Tensor]

TORCH_DTYPE_TO_OPENVINO_DTYPE = {
    torch.bool: ov.Type.boolean,
    torch.int8: ov.Type.i8,
    torch.uint8: ov.Type.u8,
    torch.int16: ov.Type.i16,
    torch.int32: ov.Type.i32,
    torch.int64: ov.Type.i64,
    torch.float32: ov.Type.f32,
    torch.float16: ov.Type.f16,
    torch.bfloat16: ov.Type.bf16,
}

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
        device_config: DeviceConfig,
        ov_core: ov.Core
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.device_config = device_config
        self.ov_core = ov_core

        self.head_size = model_config.get_head_size()
        if device_config.get_ov_device() == "CPU":
            if cache_config.cache_dtype == "u8":
                # Scale, zero point and quantized data will be stored together.
                # The layout for per token per head:
                # |scale(f32)|zeropoint(f32)|quantized data(u8,idx_1)|quantized data(u8,idx_2)|...|quantized data(u8,idx_head_size)|
                self.head_size += 8
        self.num_layers = model_config.get_num_layers(parallel_config)
        self.num_heads = model_config.get_num_kv_heads(parallel_config)

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        self.num_cpu_blocks = cache_config.num_cpu_blocks

        self.cache_dtype = OpenVINOCacheEngine.get_cache_dtype(
            self.cache_config.cache_dtype,
            self.model_config,
            self.device_config)

        # Initialize the cache.
        self.gpu_cache = self.allocate_gpu_cache()
        self.cpu_cache = self.allocate_cpu_cache()

    def get_key_block_shape(self) -> Tuple[int, int, int, int]:
        # TODO: ilavreno: we can insert per-device KV cache shapes configurations
        # which are more optimal for plugins representations
        if self.device_config.get_ov_device() == "CPU":
            return (self.num_heads, self.block_size, self.head_size)
        element_size = self.cache_dtype.get_size()
        x = 16 // element_size
        return (
            self.num_heads,
            self.head_size // x,
            self.block_size,
            x,
        )

    def get_value_block_shape(self) -> Tuple[int, int, int]:
        # TODO: ilavreno: we can insert per-device KV cache shapes configurations
        # which are more optimal for plugins representations
        if self.device_config.get_ov_device() == "CPU":
            return (self.num_heads, self.block_size, self.head_size)
        return (
            self.num_heads,
            self.head_size,
            self.block_size,
        )

    def allocate_gpu_cache(self) -> List[OpenVINOKVCache]:
        if self.num_gpu_blocks == 0:
            return None
        gpu_cache: List[OpenVINOKVCache] = []
        key_block_shape = (self.num_gpu_blocks, *self.get_key_block_shape())
        value_block_shape = (self.num_gpu_blocks, *self.get_value_block_shape())

        ov_device = self.device_config.get_ov_device()
        if "CPU" in ov_device:
            raise RuntimeError("Attempt to allocate GPU's KV cache using CPU device")

        remote_context = self.ov_core.get_default_context(ov_device)
        for _ in range(self.num_layers):
            key_blocks = remote_context.create_tensor(self.cache_dtype, ov.Shape(key_block_shape), {})
            value_blocks = remote_context.create_tensor(self.cache_dtype, ov.Shape(value_block_shape), {})
            gpu_cache.append((key_blocks, value_blocks))
        return gpu_cache

    def allocate_cpu_cache(self) -> List[OpenVINOKVCache]:
        cpu_cache: List[OpenVINOKVCache] = []
        key_block_shape = (self.num_cpu_blocks, *self.get_key_block_shape())
        value_block_shape = (self.num_cpu_blocks, *self.get_value_block_shape())
        for _ in range(self.num_layers):
            key_blocks = ov.Tensor(self.cache_dtype, key_block_shape)
            value_blocks = ov.Tensor(self.cache_dtype, value_block_shape)
            # force allocation
            key_blocks.data[:] = 0
            value_blocks.data[:] = 0
            cpu_cache.append((key_blocks, value_blocks))
        return cpu_cache

    def _swap(
        self,
        src: List[OpenVINOKVCache],
        dst: List[OpenVINOKVCache],
        src_to_dst: Dict[int, int],
    ) -> None:
        # TODO: ilavreno: implement cache sync via OpenVINO RemoteContext API host <-> device
        assert False

    def swap_in(self, src_to_dst: Dict[int, int]) -> None:
        self._swap(self.cpu_cache, self.gpu_cache, src_to_dst)

    def swap_out(self, src_to_dst: Dict[int, int]) -> None:
        self._swap(self.gpu_cache, self.cpu_cache, src_to_dst)

    def copy(self, src_to_dsts: Dict[int, List[int]]) -> None:
        if self.device_config.get_ov_device() == "CPU":
            # TODO: cache_ops.copy_blocks for CPU
            for src, dsts in src_to_dsts.items():
                for dst in dsts:
                    for layer in range(len(self.cpu_cache)):
                        self.cpu_cache[layer][0].data[dst, :] = self.cpu_cache[layer][0].data[src, :]
                        self.cpu_cache[layer][1].data[dst, :] = self.cpu_cache[layer][1].data[src, :]
        else:
            key_caches = [key_cache for key_cache, _ in self.gpu_cache]
            value_caches = [value_cache for _, value_cache in self.gpu_cache]
            # TODO: ilavreno: implement cache sync via OpenVINO RemoteContext API device -> device copies
            # cache_ops.copy_blocks(key_caches, value_caches, src_to_dsts)

    @staticmethod
    def get_cache_dtype(
        cache_dtype: str,
        model_config: ModelConfig,
        device_config: DeviceConfig) -> ov.Type:
        # TODO: ilavreno: select kv_cache data type for OpenVINO devices
        # probably, we need to force OpenVINO kv cache data types per device and assert
        # if user specified a different value
        if cache_dtype == "auto":
            ov_device = device_config.get_ov_device()
            core = ov.Core()
            inference_precision = core.get_property(ov_device, hints.inference_precision)
            if "CPU" in ov_device:
                if inference_precision == ov.Type.bf16:
                    cache_dtype = torch.bfloat16
                else:
                    cache_dtype = torch.float16
            elif "GPU" in ov_device:
                if inference_precision == ov.Type.f16:
                    cache_dtype = torch.float16
                else:
                    cache_dtype = torch.float32
            else:
                cache_dtype = model_config.dtype
        else:
            cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
        ov_cache_dtype = TORCH_DTYPE_TO_OPENVINO_DTYPE[cache_dtype]
        return ov_cache_dtype

    @staticmethod
    def get_cache_block_size(
        block_size: int,
        cache_dtype: str,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        device_config: DeviceConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        if device_config.get_ov_device() == "CPU":
            if cache_dtype == "u8":
                # Scale, zero point and quantized data will be stored together.
                # The layout for per token per head:
                # |scale(f32)|zeropoint(f32)|quantized data(u8,idx_1)|quantized data(u8,idx_2)|...|quantized data(u8,idx_head_size)|
                head_size += 8
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_layers = model_config.get_num_layers(parallel_config)

        key_cache_block = block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_layers * (key_cache_block + value_cache_block)
        cache_dtype = OpenVINOCacheEngine.get_cache_dtype(cache_dtype, model_config, device_config)
        return cache_dtype.get_size() * total


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
        # TODO: ilavreno: here we also need to measure current amount of available GPU memory in case of GPU inference
        # it will later be used to understand real
        self.init_gpu_memory = 0

    def load_model(self):
        self.model_runner.load_model()

    def profile_num_available_blocks(
        self,
        block_size: int,
        device_memory_utilization: float,
        cpu_swap_space: int,
        cache_dtype: str,
    ) -> Tuple[int, int]:
        """Profiles the peak memory usage of the model and returns the maximum
        number of Accelerator and CPU cache blocks that can be allocated.

        Args:
            block_size: The size of the cache block.
            device_memory_utilization: The fraction of the total device memory to use.
            cpu_swap_space: The size of the CPU swap space in bytes.
        """
        num_gpu_blocks, num_cpu_blocks = 0, 0
        cache_block_size = OpenVINOCacheEngine.get_cache_block_size(
            block_size, cache_dtype, self.model_config, self.parallel_config, self.device_config)

        num_cpu_blocks = int(cpu_swap_space // cache_block_size)
        num_cpu_blocks = max(num_cpu_blocks, 0)

        # TODO: ilavreno: currently we cannot specify GPU as device
        # so, the code below is dead, but we can change the condition and test it
        if "GPU" in self.device_config.get_ov_device():
            import openvino.properties.intel_gpu as intel_gpu
            import openvino.properties.device as device
            ov_device = self.device_config.get_ov_device()
            ov_core = self.model_runner.model._ov_core

            # Execute a forward pass with dummy inputs to profile the memory usage
            # of the model.
            def model_profile_run():
                # Enable top-k sampling to reflect the accurate memory usage.
                sampling_params = SamplingParams(top_p=0.99, top_k=self.vocab_size - 1)

                max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
                max_num_seqs = self.scheduler_config.max_num_seqs
                cache_config = CacheConfig(block_size, device_memory_utilization, cpu_swap_space, cache_dtype)
                cache_config.num_gpu_blocks = 1
                cache_config.num_cpu_blocks = 0

                profiling_cache_engine = OpenVINOCacheEngine(
                    cache_config,
                    self.model_config,
                    self.parallel_config,
                    self.device_config,
                    ov_core)

                # Profile memory usage with max_num_sequences sequences and the total
                # number of tokens equal to max_num_batched_tokens.
                seqs: List[SequenceGroupMetadata] = []
                for group_id in range(max_num_seqs):
                    seq_len = (max_num_batched_tokens // max_num_seqs +
                            (group_id < max_num_batched_tokens % max_num_seqs))
                    seq_data = SequenceData([0] * seq_len)
                    seq_num_blocks = (seq_len + block_size - 1) // block_size
                    block_tables = [[0] * seq_num_blocks] * max_num_seqs
                    seq = SequenceGroupMetadata(
                        request_id=str(group_id),
                        is_prompt=True,
                        seq_data={group_id: seq_data},
                        sampling_params=sampling_params,
                        block_tables=block_tables,
                        lora_request=None,
                    )
                    seqs.append(seq)

                self.model_runner.block_size = cache_config.block_size

                # Run the model with the dummy inputs.
                self.model_runner.execute_model(seqs, profiling_cache_engine.gpu_cache)

                # explicitly delete temporary KV cache manager to free KV cache when real inputs will be passed to OV
                del profiling_cache_engine

            print(f"Start profiling run with dummy inputs to evaluate memory usage for {ov_device}. It might take a while.")

            memory_statistics_before = ov_core.get_property(ov_device, intel_gpu.memory_statistics)
            model_profile_run()
            memory_statistics_after = ov_core.get_property(ov_device, intel_gpu.memory_statistics)

            used_device_mem = 0
            kv_cache_mem = 0
            # `cl_mem` buffers are used for kv_cache inputs
            if "cl_mem" in memory_statistics_before:
                kv_cache_mem = memory_statistics_after["cl_mem"] - memory_statistics_before["cl_mem"]
            else:
                kv_cache_mem = memory_statistics_after["cl_mem"]

            # sum up all used memory
            if "cl_mem" in memory_statistics_after:
                used_device_mem += memory_statistics_after["cl_mem"]
            if "usm_device" in memory_statistics_after:
                used_device_mem += memory_statistics_after["usm_device"]

            gpu_device_type = ov_core.get_property(ov_device, device.type)
            if gpu_device_type == device.Type.INTEGRATED and "usm_host" in memory_statistics_after:
                used_device_mem += memory_statistics_after["usm_host"]

            used_device_mem -= kv_cache_mem
            total_gpu_memory = ov_core.get_property(ov_device, intel_gpu.device_total_mem_size)

            print(f"Total {ov_device} memory: {total_gpu_memory} bytes. "
                  f"Amount of memory required to run the model with max_num_batched_tokens={self.scheduler_config.max_num_batched_tokens}: {used_device_mem} bytes")

            if used_device_mem >= total_gpu_memory:
                raise RuntimeError(
                    f"The required memory size {used_device_mem} bytes for model is higher than total available GPU memory {total_gpu_memory} bytes."
                    " Please consider to decrease `max_num_batched_tokens` or increase `gpu_memory_utilization`")

            cache_block_size = self.get_cache_block_size_bytes(
                block_size, cache_dtype)
            num_gpu_blocks = int(
                (total_gpu_memory * device_memory_utilization - used_device_mem) //
                cache_block_size)
            num_gpu_blocks = max(num_gpu_blocks, 0)

            if num_gpu_blocks == 0:
                raise RuntimeError(
                    "Predicted maximum of GPU KV cache blocks number is 0. Please consider to decrease `max_num_batched_tokens` or increase `gpu_memory_utilization`")

            gc.collect()

        return num_gpu_blocks, num_cpu_blocks

    def init_cache_engine(self, cache_config: CacheConfig) -> None:
        self.cache_config = cache_config
        self.cache_engine = OpenVINOCacheEngine(self.cache_config, self.model_config,
                                                self.parallel_config, self.device_config,
                                                self.model_runner.model._ov_core)
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

        # cache_events = self.cache_events if issued_cache_op else None

        # Wait for cache operations to finish.
        # TODO: ilavrenov, probably, we need similar for GPU if we can schedule multiple operations
        # and then call a single syncronization
        # if cache_events is not None:
        #     for event in cache_events:
        #         event.wait()

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

        device_cache = self.gpu_cache if self.cache_config.num_gpu_blocks > 0 else self.cpu_cache
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
                                                        self.parallel_config,
                                                        self.device_config)

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
        if os.environ.get("VLLM_OPENVINO_CPU_KV_CACHE_PRECISION", "") == "u8":
            cache_config.cache_dtype = "u8"
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
            kv_cache_dtype=OpenVINOCacheEngine.get_cache_dtype(self.cache_config.cache_dtype,
                                                               self.model_config,
                                                               self.device_config)
        )
        self.driver_worker.init_model()
        self.driver_worker.load_model()

    def _init_cache(self) -> None:
        """Profiles the memory usage and initializes the KV cache.

        The engine first profiles the existing memory usage.
        Then, it allocates the remaining memory for KV blocks.

        .. tip::
            You may limit the usage of device memory
            by adjusting the `device_memory_utilization` parameter.
        """
        # Get the maximum number of blocks that can be allocated on GPU and CPU.
        num_gpu_blocks, num_cpu_blocks = (
            self.driver_worker.profile_num_available_blocks(
                block_size=self.cache_config.block_size,
                device_memory_utilization=self.cache_config.gpu_memory_utilization,
                cpu_swap_space=self.cache_config.swap_space_bytes,
                cache_dtype=self.cache_config.cache_dtype,
            ))

        logger.info(f"# GPU blocks: {num_gpu_blocks}, "
                    f"# CPU blocks: {num_cpu_blocks}")

        num_blocks = num_gpu_blocks if num_gpu_blocks > 0 else num_cpu_blocks
        check_block_size_valid(num_blocks, self.cache_config.block_size,
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
