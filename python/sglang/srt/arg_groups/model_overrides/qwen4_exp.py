"""Config-time override declarations for qwen4_exp.

Architectures: Qwen4ExpForConditionalGeneration

Chuyen tu ban inline cua PR #36497 sang package model_overrides/ ma upstream
gioi thieu sau do. CANH BAO: module nay VA qwen3_5.py cung khai `page_size` cho kien truc nay.
Package cam hai module khai cung mot truong, nhung validate_declarations chi
kiem whitelist TEN nen khong bat duoc. Tren H200/SM90 vo hai vi
_qwen3_5_hybrid_overrides return {} ngay (no chi chay tren SM100). Neu chay
tren SM100 thi phai tach lai: quyet dinh ai so huu page_size cho Qwen4-Exp.
"""

import logging
from typing import Any, Dict

from sglang.srt.arg_groups.model_override_base import _register_for, model_config_of
from sglang.srt.utils import is_cuda

logger = logging.getLogger(__name__)


@_register_for("Qwen4ExpForConditionalGeneration")
def _qwen4_exp_overrides(server_args: Any, hf_config: Any) -> dict:
    """Qwen4-Exp keeps the MoE config under ``text_config``; every layer is
    sparse, so a dense-MLP TP size of 1 only stalls the DP MoE path.

    Compressed QSA additionally pins page_size=64 (overriding the hybrid
    family's triton default of 1, last-writer-wins): its compressed cache is
    addressed as ``full_slot // compress_ratio`` (the DSV4 scheme), which
    requires page-aligned full-KV allocation with the page a multiple of the
    compress ratio, and page-granular prefix sharing so shared pages share
    their compressed slots. MambaRadixCache supports page_size > 1 only with
    the mamba extra-buffer strategy, so fall back to the family default when
    neither that nor --disable-radix-cache holds (the QSA pool then fails
    fast at boot).
    """
    overrides: Dict[str, Any] = {}
    if server_args.ple_offload_embedding is None:
        import torch

        overrides["ple_offload_embedding"] = (
            is_cuda() and model_config_of(server_args).dtype == torch.bfloat16
        )

    text_config = getattr(hf_config, "text_config", hf_config)
    if (
        getattr(text_config, "num_experts", None) is not None
        and server_args.moe_dense_tp_size == 1
    ):
        overrides["moe_dense_tp_size"] = None

    from sglang.srt.layers.attention.qsa.config import (
        QSA_VARIANT_COMPRESSED,
        parse_qsa_profile,
    )

    profile = parse_qsa_profile(hf_config)
    if profile is not None and profile.variant == QSA_VARIANT_COMPRESSED:
        # Unconditional, like DeepSeek-V4's page-256 declaration: compressed
        # addressing is full_slot // ratio and requires page-aligned
        # allocation on every backend.
        overrides["page_size"] = 64
        logger.info(
            "Setting page size to 64 for compressed QSA "
            "(full//ratio compressed addressing)."
        )
    return overrides
