from dataclasses import dataclass
from typing import List, Tuple, Type

import openvino as ov
import torch

import vllm.envs as envs
from vllm.attention.backends.abstract import (AttentionBackend,
                                              AttentionMetadata)
from vllm.attention.backends.utils import CommonAttentionState


class OpenVINOAttentionBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "openvino"

    @staticmethod
    def get_impl_cls():
        # OpenVINO implements PagedAttention as part of the Optimum
        # exported model
        raise NotImplementedError

    @staticmethod
    def make_metadata(*args, **kwargs) -> "AttentionMetadata":
        raise NotImplementedError

    @staticmethod
    def get_state_cls() -> Type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def make_openvino_metadata(*args, **kwargs) -> "OpenVINOAttentionMetadata":
        return OpenVINOAttentionMetadata(*args, **kwargs)

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        if "CPU" in envs.VLLM_OPENVINO_DEVICE:
            return (2, num_blocks, num_kv_heads, block_size, head_size)
        else:
            return (2, num_blocks, num_kv_heads, head_size, block_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: ov.Tensor,
        dst_kv_cache: ov.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        begin_roi = ov.runtime.Coordinate([0, 0, 0, 0])
        end_roi = ov.runtime.Coordinate(src_kv_cache.get_shape())

        def get_roi_tensor(tensor, block_number):
            roi_begin = ov.runtime.Coordinate(begin_roi)
            roi_end = ov.runtime.Coordinate(end_roi)
            roi_begin[0] = block_number
            roi_end[0] = block_number + 1

            if isinstance(tensor, ov.Tensor):
                return ov.Tensor(tensor, roi_begin, roi_end)
            else:
                return ov.RemoteTensor(tensor, roi_begin, roi_end)

        num_blocks = src_to_dst.size(0)
        for i in range(num_blocks):
            src_block_number = src_to_dst[i, 0].item()
            dst_block_number = src_to_dst[i, 1].item()

            src_roi_tensor = get_roi_tensor(src_kv_cache, src_block_number)
            dst_roi_tensor = get_roi_tensor(dst_kv_cache, dst_block_number)
            src_roi_tensor.copy_to(dst_roi_tensor)


    @staticmethod
    def copy_blocks(
        kv_caches: List[Tuple[ov.Tensor, ov.Tensor]],
        src_to_dists: List[Tuple[int, int]],
    ) -> None:
        for src, dst in src_to_dists:
            for key_cache, value_cache in kv_caches:
                key_cache.data[dst, :] = key_cache.data[src, :]
                value_cache.data[dst, :] = value_cache.data[src, :]


@dataclass
class OpenVINOAttentionMetadata:
    """Metadata for OpenVINOAttentionBackend.

    Basic terms used below:
    - batch_size_in_sequences - total number of sequences to execute​
    - prompt_lens – per sequence size number of scheduled tokens​
    - batch_size_in_tokens = sum(prompt_lens)​
    - max_context_len = max(context_lens)​
    - max_num_blocks = div_up(max_context_len / BLOCK_SIZE)​
    - num_blocks – total number of blocks in block_indices​
    """

    # Describes past KV cache size for each sequence within a batch
    # Shape: [batch_size_in_sequences]
    # Type: i32​
    past_lens: torch.Tensor

    # Describes start indices of input / speculative tokens from
    # current sequences within a batch sequence​
    # Shape: [batch_size_in_sequences + 1]​
    # Type: i32
    subsequence_begins: torch.Tensor

    # Describes block tables for each sequence within a batch​ -
    # indices along 0th dimension in key_cache and value_cache inputs​
    # Shape: [num_blocks]
    # Type: i32​
    block_indices: torch.Tensor

    # Describes block tables for each sequence within a batch​ -
    # for i-th element, it is an index in block_indices with the
    # first block belonging to i-th sequence​
    # Shape: [batch_size_in_sequences + 1]
    # Type: i32​
    block_indices_begins: torch.Tensor

    # Describes max context length
    # Shape: scalar
    # Type: i32
    max_context_len: torch.Tensor
