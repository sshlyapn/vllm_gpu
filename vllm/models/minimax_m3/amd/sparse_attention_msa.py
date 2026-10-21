# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MSA AITER block-sparse attend for MiniMax M3."""

import os
from collections import Counter
from dataclasses import dataclass

import torch

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.minimax_m3.common.sparse_attention import (
    MiniMaxM3SparseBackend,
    MiniMaxM3SparseDecodeMetadata,
    MiniMaxM3SparseImpl,
    MiniMaxM3SparseMetadata,
    MiniMaxM3SparseMetadataBuilder,
    MiniMaxM3SparsePrefillMetadata,
)
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (
    AttentionLayer,
)
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

# Debug: histogram of (prefill rows, dqC groups) per metadata build, logged
# every _DQC_LOG_EVERY prefill builds.
_DQC_LOG_GROUPS = os.environ.get("MINIMAX_M3_DQC_LOG_GROUPS", "0") == "1"
_DQC_LOG_EVERY = 10
_dqc_group_hist: Counter[tuple[int, int]] = Counter()


def _log_dqc_groups(rows: int, groups: int) -> None:
    _dqc_group_hist[(rows, groups)] += 1
    total = sum(_dqc_group_hist.values())
    if total % _DQC_LOG_EVERY == 0:
        hist = " ".join(
            f"{r}r{g}g:{c}" for (r, g), c in sorted(_dqc_group_hist.items())
        )
        logger.info("MiniMax-M3 dqC groups after %d prefill builds: %s", total, hist)


class MiniMaxM3SparseAiterPABackend(MiniMaxM3SparseBackend):
    """MiniMax M3 backend carrying the AITER page-16 block-table rebase."""

    @staticmethod
    def get_builder_cls() -> type["MiniMaxM3SparseAiterPAMetadataBuilder"]:
        return MiniMaxM3SparseAiterPAMetadataBuilder


@dataclass
class MiniMaxM3SparseAiterPAPrefillMetadata(MiniMaxM3SparsePrefillMetadata):
    # ``block_table`` rebased onto AITER's page-16 numbering; see
    # ``MiniMaxM3SparseAiterPAMetadataBuilder.build``.
    page16_block_table: torch.Tensor | None = None
    # dqC row groups that fit the reserved workspace; see minimax_m3_dqc_groups.
    dqc_groups: list[tuple[int, int, int, int, int]] | None = None


@dataclass
class MiniMaxM3SparseAiterPADecodeMetadata(MiniMaxM3SparseDecodeMetadata):
    page16_block_table: torch.Tensor | None = None


class MiniMaxM3SparseAiterPAMetadataBuilder(MiniMaxM3SparseMetadataBuilder):
    """Adds the page-16 rebase of the block table the indexer's top-k needs.

    The AITER indexer's top-k emits the attend's page table, expanding a
    selected block into a compile-time number of pages -- one side's worth,
    while an interleaved block holds both -- so it has to resolve the selection
    through a table pre-scaled to the wider stride. Every sparse layer selects
    through the same table, so it is rebased here once per step rather than
    once per layer, alongside the slot mapping the base rebases for the writer.
    """

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        from vllm.models.minimax_m3.amd.ops.sparse_pa import (
            minimax_m3_dqc_budget_blocks,
        )

        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # Stable address for the same reason as the base's own buffers: a
        # captured decode replay writes through the address it captured.
        self.page16_block_table_buffer = torch.empty(
            (
                vllm_config.scheduler_config.max_num_seqs,
                cdiv(vllm_config.model_config.max_model_len, kv_cache_spec.block_size),
            ),
            dtype=torch.int32,
            device=device,
        )
        self.dqc_budget_blocks = minimax_m3_dqc_budget_blocks(
            vllm_config.model_config.max_model_len
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MiniMaxM3SparseMetadata:
        from vllm.models.minimax_m3.amd.ops.sparse_pa import (
            minimax_m3_dqc_enabled,
            minimax_m3_dqc_groups,
            minimax_m3_rebase_block_table_to_page16,
        )

        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        block_table = common_attn_metadata.block_table_tensor
        num_reqs, cols = block_table.shape
        assert (
            num_reqs <= self.page16_block_table_buffer.shape[0]
            and cols <= self.page16_block_table_buffer.shape[1]
        ), (
            f"block table {(num_reqs, cols)} exceeds the page-16 rebase buffer "
            f"{tuple(self.page16_block_table_buffer.shape)}"
        )
        page16 = minimax_m3_rebase_block_table_to_page16(
            block_table,
            out=self.page16_block_table_buffer[:num_reqs, :cols],
        )

        # Decode-first batch, so each side takes the rows its own block table
        # was sliced from. Reconstructed from the base's fields rather than
        # listed out, so a field added there does not silently drop here.
        nd = metadata.num_decodes
        if metadata.decode is not None:
            metadata.decode = MiniMaxM3SparseAiterPADecodeMetadata(
                **vars(metadata.decode),
                page16_block_table=page16[:nd],
            )
        if metadata.prefill is not None:
            dqc_groups = None
            if minimax_m3_dqc_enabled():
                n = common_attn_metadata.num_reqs
                seq_ub = common_attn_metadata.seq_lens_cpu_upper_bound
                assert seq_ub is not None
                qsl = common_attn_metadata.query_start_loc_cpu[nd : n + 1]
                dqc_groups = minimax_m3_dqc_groups(
                    seq_ub[nd:n].tolist(),
                    (qsl - qsl[0]).tolist(),
                    self.dqc_budget_blocks,
                )
                if _DQC_LOG_GROUPS:
                    _log_dqc_groups(n - nd, len(dqc_groups))
            metadata.prefill = MiniMaxM3SparseAiterPAPrefillMetadata(
                **vars(metadata.prefill),
                page16_block_table=page16[nd:],
                dqc_groups=dqc_groups,
            )
        return metadata


class MiniMaxM3SparseAiterPAImpl(MiniMaxM3SparseImpl):
    """ROCm AITER page-16 SHUFFLE sparse paged attention."""

    def __init__(self, *args, **kwargs) -> None:
        from vllm.models.minimax_m3.amd.ops.sparse_pa import (
            PAGES_PER_SPARSE_BLOCK,
            minimax_m3_dqc_budget_blocks,
            minimax_m3_dqc_enabled,
        )

        super().__init__(*args, **kwargs)
        # The dqC fp8 workspace is taken from vLLM's workspace manager at its
        # full size on the first forward (the memory profiling run), so it is
        # counted before the KV cache is sized and never grows after lock.
        self.dqc_ws_pages = 0
        if minimax_m3_dqc_enabled():
            max_model_len = get_current_vllm_config().model_config.max_model_len
            self.dqc_ws_pages = (
                minimax_m3_dqc_budget_blocks(max_model_len) * PAGES_PER_SPARSE_BLOCK
            )
        self._dqc_reserved = False

    def _reserve_dqc_workspace(self) -> None:
        from vllm.models.minimax_m3.amd.ops.sparse_pa import (
            minimax_m3_dqc_reserve_rows,
            minimax_m3_dqc_workspace,
        )

        k8, v8 = minimax_m3_dqc_workspace(self.dqc_ws_pages)
        self._dqc_reserved = True
        logger.info_once(
            "MiniMax-M3 dqC: reserved %.1f MiB of workspace (%d rows x "
            "max_model_len)",
            (k8.numel() + v8.numel()) / 2**20,
            minimax_m3_dqc_reserve_rows(),
        )

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        *,
        query_fp8: torch.Tensor | None = None,
        decode_sparse_table: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        from vllm.models.minimax_m3.amd.ops.sparse_pa import (
            PAGES_PER_SPARSE_BLOCK,
            minimax_m3_page16_sides_per_block,
            minimax_m3_sparse_attn_decode_aiter,
            minimax_m3_sparse_attn_prefill_aiter,
            minimax_m3_sparse_block_page_stride,
        )

        if self.dqc_ws_pages and not self._dqc_reserved:
            self._reserve_dqc_workspace()
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return output
        main_md = attn_metadata[layer.layer_name]  # type: ignore[attr-defined]
        assert isinstance(main_md, MiniMaxM3SparseMetadata)

        nd = main_md.num_decode_tokens
        num_tokens = main_md.num_actual_tokens
        topk = layer.topk_indices_buffer  # type: ignore[attr-defined]
        assert topk is not None
        # Set only when the layer's indexer emitted the page table alongside its
        # selection; otherwise these stay None and the table is built below.
        sparse_bt_buf = getattr(layer, "sparse_bt_buffer", None)
        sparse_ctx_buf = getattr(layer, "sparse_ctx_buffer", None)
        kvh = self.num_kv_heads
        if kvh != 1 and sparse_bt_buf is None:
            raise NotImplementedError(
                "MiniMax-M3 AITER sparse PA needs the page table the indexer's "
                f"top-k emits to serve per-rank num_kv_heads == {kvh}; the "
                "Triton builders it falls back to address one head's cache."
            )

        hd = self.head_size
        q = query[:num_tokens].view(-1, self.num_heads, hd)
        out = output[:num_tokens].view(-1, self.num_heads, hd)
        k_cache, v_cache = layer.get_aiter_sparse_pa_kv_cache()  # type: ignore[attr-defined]
        k_scale = getattr(layer, "_k_scale", None) if self.use_fp8_kv else None
        v_scale = getattr(layer, "_v_scale", None) if self.use_fp8_kv else None

        if sparse_bt_buf is not None:
            # An emitted table was scaled by the metadata builder, which reads
            # the packing off the layout rather than off these tensors. The two
            # only disagree if the resolved layout gives each side its own
            # plane, and a silent disagreement reads unrelated pages.
            stride = minimax_m3_sparse_block_page_stride(k_cache, v_cache)
            expected = PAGES_PER_SPARSE_BLOCK * minimax_m3_page16_sides_per_block()
            if stride != expected:
                raise RuntimeError(
                    "MiniMax-M3 AITER sparse PA: the indexer emitted a page "
                    f"table for a block spanning {expected} pages, but this "
                    f"KV cache lays a block out over {stride}. The page-16 "
                    "rebase assumes both K/V sides share a block."
                )

        if main_md.num_decodes > 0:
            d = main_md.decode
            assert d is not None
            prepared_decode_table = decode_sparse_table
            if sparse_bt_buf is not None:
                assert sparse_ctx_buf is not None
                prepared_decode_table = (
                    sparse_bt_buf[: nd * kvh],
                    sparse_ctx_buf[: nd * kvh],
                )
            minimax_m3_sparse_attn_decode_aiter(
                q[:nd],
                k_cache,
                v_cache,
                topk[:, :nd, :],
                d.block_table,
                d.seq_lens,
                self.num_kv_heads,
                self.scale,
                out[:nd],
                k_scale=k_scale,
                v_scale=v_scale,
                decode_query_len=d.decode_query_len,
                sparse_block_table=(
                    prepared_decode_table[0]
                    if prepared_decode_table is not None
                    else None
                ),
                sparse_context_lens=(
                    prepared_decode_table[1]
                    if prepared_decode_table is not None
                    else None
                ),
            )

        if main_md.num_prefills > 0:
            p = main_md.prefill
            assert p is not None
            assert p.query_req_id is not None and p.query_abs_pos is not None
            minimax_m3_sparse_attn_prefill_aiter(
                q[nd:],
                k_cache,
                v_cache,
                topk[:, nd:num_tokens, :],
                p.block_table,
                p.query_req_id,
                p.query_abs_pos,
                self.num_kv_heads,
                self.scale,
                out[nd:],
                k_scale=k_scale,
                v_scale=v_scale,
                sparse_bt=(
                    None
                    if sparse_bt_buf is None
                    else sparse_bt_buf[nd * kvh : num_tokens * kvh]
                ),
                sparse_ctx=(
                    None
                    if sparse_ctx_buf is None
                    else sparse_ctx_buf[nd * kvh : num_tokens * kvh]
                ),
                seq_lens=p.seq_lens,
                max_seq_len=p.max_seq_len,
                total_kv_blocks=p.total_kv_blocks,
                dqc_groups=p.dqc_groups,
                dqc_ws_pages=self.dqc_ws_pages or None,
            )
        return output
