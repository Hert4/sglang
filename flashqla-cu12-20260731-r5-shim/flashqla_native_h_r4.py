#!/usr/bin/env python3
"""r4: FlashQLA extend TU SINH mid-chunk states `h` — het fallback, an duoc
extra_buffer o moi prefill dai (CCR / long-context).

Nen tang (probe 31/07 tren H200, container r3):
  flash_qla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule_fwd(output_h=True)
  tra (g, A, o, h, final_state, cp_cache); voi input GDN Qwen3.6 (Hq16/HV32/K128/V128,
  varlen 2 seq 20480+12288):
    - h shape (1, 512, 32, 128, 128) = Y HET fork sglang, 64 token/chunk
    - auto_cp map h ve layout varlen goc NGAY TRONG thu vien
    - parity vs fork:  o maxdiff 1.7e-3 · h same-layout mean 2.2e-5
      (layout transposed sai gap 250x => khong co nham lan K/V)
    - final_state same-layout 5e-4
  l2norm ap NGOAI bang flash_qla.utils.l2norm_fwd (tang fwd khong co co in-kernel).

Ap tren wrapper DA CO r3 (track_needs_h da duoc call-site truyen xuong).
Fail-closed: moi phep sua assert khop dung 1 vi tri.
"""
import sys
from pathlib import Path

SGL = Path(sys.argv[1]) if len(sys.argv) > 1 else None
assert SGL and SGL.is_dir(), f"usage: {sys.argv[0]} <sglang-pkg-dir>; got {SGL}"
F = SGL / "srt/layers/attention/linear/kernels/gdn_flashqla.py"


def patch_once(old: str, new: str) -> None:
    src = F.read_text()
    n = src.count(old)
    assert n == 1, f"[r4] pattern khop {n} lan (can 1):\n{old!r}"
    F.write_text(src.replace(old, new))


# 1) Loader: them API tang thap + l2norm
patch_once(
    """def _load_flash_qla():""",
    """def _load_flash_qla_lowlevel():
    \"\"\"Import (fwd tang thap, l2norm) cho duong native-h (cached).\"\"\"
    global _flash_qla_lowlevel
    if _flash_qla_lowlevel is None:
        from flash_qla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd
        from flash_qla.utils import l2norm_fwd

        _flash_qla_lowlevel = (chunk_gated_delta_rule_fwd, l2norm_fwd)
    return _flash_qla_lowlevel


def _load_flash_qla():""",
)
patch_once(
    """_flash_qla_chunk = None""",
    """_flash_qla_chunk = None
_flash_qla_lowlevel = None""",
)

# 2) Bo fallback theo track_needs_h (r3) — thay bang duong native-h.
#    Giu _should_fall_back(q) cho nguong kich thuoc/dtype.
patch_once(
    """        if track_needs_h:
            _log_track_fallback_once()
        if track_needs_h or self._should_fall_back(q):""",
    """        if self._should_fall_back(q):""",
)

# 3) Duong native-h: khi track can h, dung API tang thap
patch_once(
    """        chunk_gated_delta_rule = _load_flash_qla()""",
    """        if track_needs_h:
            # extra_buffer can mid-chunk states: dung API tang thap output_h=True.
            # h tra ve dung layout/granularity cua fork (probe 31/07) nen dua
            # thang cho _track_mamba_state_extend, khong can map.
            _log_track_native_once()
            fwd, l2norm = _load_flash_qla_lowlevel()
            qn, _ = l2norm(q.contiguous())
            kn, _ = l2norm(k.contiguous())
            initial_state = ssm_states[cache_indices].to(q.dtype)
            ret = fwd(
                q=qn,
                k=kn,
                v=v.contiguous(),
                g=g.contiguous(),
                beta=beta.contiguous(),
                initial_state=initial_state,
                # copy=True (tu build sau r5): phong thu truong hop upstream doi
                # query_start_loc sang buffer int64 persistent — .to(long) khi do
                # tra ve CHINH tensor cu, tensor_cache cua flash_qla key theo
                # identity se hit stale -> sai cu_seqlens am tham. Base 20260731
                # cap int32 moi moi forward nen r5 da ship van an toan.
                cu_seqlens=query_start_loc.to(torch.long, copy=True),
                output_final_state=True,
                output_h=True,
                state_v_first=True,
                # r5: khop default cua high-level API. AUDIT 31/07 doc source:
                # co nay KHONG phai cache cross-forward — no chon bien the kernel
                # get_warmup_chunks (single) vs _bidi; high-level luon dung bidi
                # (da duoc JIT-warm), con single la kernel TileLang khac chua warm
                # => nghi la nguon +205% cold TTFT cua r4. So do truoc/sau la that,
                # co che o muc gia-thuyet-khop-nhat.
                enable_fwd_cp_cache=True,
            )
            # (g, A, o, h, final_state, cp_cache) tren flash-qla 0.1.2 — neu
            # upstream doi arity thi tha loudly con hon lay nham tensor.
            assert len(ret) == 6, f"flash_qla fwd arity {len(ret)} != 6"
            o, h, final_state = ret[2], ret[3], ret[4]
            ssm_states[cache_indices] = final_state.to(ssm_states.dtype)
            return o, None, h

        chunk_gated_delta_rule = _load_flash_qla()""",
)

# 4) Doi log fallback -> log native (giu ham cu de khoi vo import o cho khac)
patch_once(
    """_track_fallback_logged = False""",
    """_track_fallback_logged = False
_track_native_logged = False


def _log_track_native_once():
    global _track_native_logged
    if not _track_native_logged:
        _track_native_logged = True
        import logging

        logging.getLogger(__name__).info(
            "FlashQLA extend: mamba radix-track can mid-chunk states (h); "
            "dung duong native output_h=True (khong fallback)."
        )""",
)

import py_compile

py_compile.compile(str(F), doraise=True)
src = F.read_text()
assert "output_h=True" in src and "_load_flash_qla_lowlevel" in src
# r3 fallback theo track khong con duoc GOI — chuoi chi con xuat hien o dong `def`
n_call = src.count("_log_track_fallback_once()")
assert n_call == 1, f"_log_track_fallback_once() xuat hien {n_call} lan (mong 1 = dong def)"
assert "track_needs_h or self._should_fall_back" not in src, "van con dieu kien fallback r3"
print("[r4+r5] OK: native-h + fwd_cp_cache, py_compile qua")
