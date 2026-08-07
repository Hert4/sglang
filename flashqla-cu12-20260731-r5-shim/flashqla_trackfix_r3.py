#!/usr/bin/env python3
"""r3: FlashQLA extend + mamba extra_buffer track — fallback Triton khi can `h`.

Bug (do 31/07 tren hackathon, image r2): FlashQLA `chunk_gated_delta_rule` chi tra
final_state, KHONG co checkpoint giua chunk (`h`). Khi mamba radix cache
(extra_buffer) can luu state tai ranh gioi track nam TRONG chunk dang prefill
(`forward_metadata.track_ssm_h_src.numel() > 0`), call-site doi `h`:
    hybrid_linear_attn_backend._track_mamba_state_extend: assert h is not None
=> SIGQUIT ngay forward 16384-token dau tien. Triton cung tai chay 21 chunk khong sao.

Fix: truyen co `track_needs_h` tu call-site (noi duy nhat co forward_metadata)
xuong kernel; FlashQLA thay co thi fallback super().extend() (Triton, luon tra h).

VI SAO KHONG di duong checkpoint-plan cua FlashInfer (`uses_state_checkpoints=True`):
plan builder GHI DE `track_ssm_h_src` sang layout packed cua FlashInfer; neu sau do
fallback Triton (h layout day du) thi index lech => ghi nham state, hong am tham.
Giu `uses_state_checkpoints=False` => track_ssm_h_src luon o layout Triton => khop.

Fail-closed: moi phep sua deu assert dung 1 vi tri khop; lech la build fail.
"""
import re
import sys
from pathlib import Path

SGL = Path(sys.argv[1]) if len(sys.argv) > 1 else None
assert SGL and SGL.is_dir(), f"usage: {sys.argv[0]} <sglang-pkg-dir>; got {SGL}"

LINEAR = SGL / "srt/layers/attention/linear"


def patch_once(path: Path, old: str, new: str) -> None:
    src = path.read_text()
    n = src.count(old)
    assert n == 1, f"[r3] {path.name}: pattern khop {n} lan (can dung 1):\n{old!r}"
    path.write_text(src.replace(old, new))
    print(f"[r3] OK: {path.name}")


# ---- 0) Tien de: dispatcher.extend phai chuyen **kwargs xuong extend_kernel ----
gdn_backend = LINEAR / "gdn_backend.py"
src = gdn_backend.read_text()
m = re.search(
    r"def extend\(\s*self,.*?return self\.extend_kernel\.extend\((.*?)\)",
    src,
    re.S,
)
assert m and "**kwargs" in m.group(1), (
    "[r3] GDNKernelDispatcher.extend khong forward **kwargs xuong extend_kernel — "
    "co `track_needs_h` se bi nuot. Xem lai truoc khi build."
)
print("[r3] OK: dispatcher forward **kwargs")

# ---- 1) Call-site: tinh track_needs_h tu forward_metadata ----
# Anchor vao kwarg state_checkpoint_cu_starts — chi call-site extend co no.
patch_once(
    gdn_backend,
    """                query_start_loc=query_start_loc,
                state_checkpoint_cu_starts=(
                    forward_metadata.state_checkpoint_cu_starts
                ),""",
    """                query_start_loc=query_start_loc,
                # FlashQLA khong san xuat duoc checkpoint giua chunk; bao truoc
                # cho kernel biet forward nay can `h` de no fallback Triton.
                # (numel() la metadata host-side, khong GPU sync.)
                track_needs_h=(
                    forward_metadata.has_mamba_track_mask
                    and forward_metadata.track_ssm_h_src is not None
                    and forward_metadata.track_ssm_h_src.numel() > 0
                ),
                state_checkpoint_cu_starts=(
                    forward_metadata.state_checkpoint_cu_starts
                ),""",
)

# ---- 2) Kernel wrapper: nhan co va fallback ----
gdn_flashqla = LINEAR / "kernels/gdn_flashqla.py"
patch_once(
    gdn_flashqla,
    """        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> tuple:
        if self._should_fall_back(q):""",
    """        query_start_loc: torch.Tensor,
        track_needs_h: bool = False,
        **kwargs,
    ) -> tuple:
        if track_needs_h or self._should_fall_back(q):""",
)

# Ghi log MOT lan khi fallback vi track (de ops thay flashqla dang nhuong Triton).
patch_once(
    gdn_flashqla,
    """_flash_qla_chunk = None""",
    """_flash_qla_chunk = None
_track_fallback_logged = False


def _log_track_fallback_once():
    global _track_fallback_logged
    if not _track_fallback_logged:
        _track_fallback_logged = True
        import logging

        logging.getLogger(__name__).info(
            "FlashQLA extend: mamba radix-track can mid-chunk states (h); "
            "fallback Triton cho cac forward nay (dung, cham hon)."
        )""",
)
patch_once(
    gdn_flashqla,
    """        if track_needs_h or self._should_fall_back(q):""",
    """        if track_needs_h:
            _log_track_fallback_once()
        if track_needs_h or self._should_fall_back(q):""",
)

# ---- 3) Nghiem thu cu phap ----
import py_compile

for f in (gdn_backend, gdn_flashqla):
    py_compile.compile(str(f), doraise=True)
print("[r3] OK: py_compile ca hai file")
