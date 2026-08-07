# `ductransa01/sglang-flashqla:cu12-20260731-r5-shim`

Recipe build đầy đủ của image **`ductransa01/sglang-flashqla:cu12-20260731-r5-shim`**
(digest `f5cf60cf2d8f`) — bản sglang + FlashQLA GDN-prefill đã lên production cho
Qwen3.6-35B-A3B (FP8, TP1, H200), ngày 31/07/2026.

Branch này được đặt **đúng tên tag Docker** để tra sau cho dễ:
`cu12-20260731-r5-shim`. Đây là branch **lưu recipe + báo cáo**, không phải branch
sửa source sglang: các thay đổi source nằm trong `flashqla-gdn-prefill.patch` và 2
script vá (`flashqla_trackfix_r3.py`, `flashqla_native_h_r4.py`) — chúng vá **package
sglang đã cài trong image nightly**, không vá cây source của repo này.
Base branch: `main` @ `874fc07d9bbbb714a71e5d4cbe5e005a885168ef` (13/07/2026) — không
liên quan tới nội dung image, chỉ là điểm nhánh.

## Chuỗi build (mỗi tầng FROM tầng trước)

| Tag | Dockerfile | Payload | Ghi chú |
|---|---|---|---|
| base | — | `lmsysorg/sglang:nightly-dev-cu12-20260731-68d44294` | nightly đã có fix #27998 (guard `mamba_next_track_idx`) |
| `cu12-20260731-r2` | `Dockerfile.cu12-rebase` | `flashqla-gdn-prefill.patch` + `pip install --no-deps flash-qla==0.1.2` | file này đã mang bản vá bug #1; header trong file vẫn ghi tag cũ `cu12-20260731` — tag đó là bản **trước** khi có `--no-deps`, **hỏng, đừng dùng** |
| `cu12-20260731-r3` | `Dockerfile.r3-trackfix` | `flashqla_trackfix_r3.py` | fallback Triton khi mamba radix-track cần mid-chunk `h` (bug #2) |
| `cu12-20260731-r4` | `Dockerfile.r4-native-h` | `flashqla_native_h_r4.py` | FlashQLA tự sinh mid-chunk states qua `output_h=True` |
| `cu12-20260731-r5` | `Dockerfile.r4-native-h` (build lại) | `flashqla_native_h_r4.py` (đã thêm `enable_fwd_cp_cache=True`) | **KHÔNG có `Dockerfile.r5`** — xem mục dưới |
| `cu12-20260731-r5-shim` | `shim/Dockerfile.shim-r5` | `shim/sitecustomize.py` | lớp strip `RUNAI_PATH_PREFIX` cho ingress run.ai. **Bản lên PROD** |

### Vì sao không có `Dockerfile.r5`

r5 = r4 + **một dòng** `enable_fwd_cp_cache=True` (bug #4: cờ này chọn biến thể kernel
`get_warmup_chunks_bidi` thay vì `get_warmup_chunks` chưa được JIT-warm → cold TTFT p95
+205%). Fix được sửa **trực tiếp trong `flashqla_native_h_r4.py`** (dòng ~99) rồi build
lại bằng `Dockerfile.r4-native-h` và tag thành `-r5`. File trong repo này **đã là bản
r5**; build lại theo `Dockerfile.r4-native-h` sẽ ra r5, không ra r4.

## Build lại từ đầu

```bash
cd flashqla-cu12-20260731-r5-shim

docker build -f Dockerfile.cu12-rebase  -t ductransa01/sglang-flashqla:cu12-20260731-r2 .
docker build -f Dockerfile.r3-trackfix  -t ductransa01/sglang-flashqla:cu12-20260731-r3 .
docker build -f Dockerfile.r4-native-h  -t ductransa01/sglang-flashqla:cu12-20260731-r5 .   # xem mục trên
docker build -f shim/Dockerfile.shim-r5 -t ductransa01/sglang-flashqla:cu12-20260731-r5-shim shim/
```

Nhanh hơn: image đã có trên Docker Hub, chỉ cần pull rồi retag về registry nội bộ:

```bash
docker pull ductransa01/sglang-flashqla:cu12-20260731-r5-shim   # digest f5cf60cf2d8f
```

Sau khi deploy image mới, **bắt buộc** chạy pre-warm một lần (đốt JIT TileLang vào
`TILELANG_CACHE_DIR` trên PVC, tránh p99 TTFT nhảy ~18s lúc phục vụ):

```bash
python3 prewarm_flashqla.py --base-url <GATEWAY-URL> --api-key <KEY> --model <MODEL>
# chạy 2 lần: lần 2 mọi vòng phải <15s, nếu vẫn chậm là cache PVC không ăn
```

## Tags — cái nào dùng được

- `cu12-20260731-r5-shim` (`f5cf60cf2d8f`) — **bản PROD**
- `cu12-20260731-r5` (`cddf85bb7762`) — như trên, chưa có lớp shim
- `cu12-20260731-r4` / `-r4-shim` — ⚠️ dính bug #4 (cold TTFT p95 +205%), đừng ship
- `cu12-20260731-r3` — bảo thủ: fallback Triton thay vì native-h (an toàn, lợi ích ≈ 0)
- `cu12-20260731-r2` — thiếu trackfix: flashqla + extra_buffer sập ở prefill ≥16K
- `cu12-20260731` — ⛔ hỏng (bug #1), không dùng

## Báo cáo

`BAOCAO_BENCH_3107.md` — 4 bug tìm được trong quá trình rebase + toàn bộ số benchmark
A/B (triton vs flashqla), CCR sweep, và runbook deploy.

## Đã lược bỏ khi đưa lên public

IP/hostname registry và gateway nội bộ được thay bằng placeholder `<HARBOR>`,
`<HARBOR-PROD>`, `<HARBOR-TEST>`, `<GATEWAY-HOST>`. Nội dung kỹ thuật (bug, patch, số
đo, digest) giữ nguyên 100%.
