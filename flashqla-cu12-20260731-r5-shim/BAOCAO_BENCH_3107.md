# Báo cáo 31/07: sglang vs sglang+FlashQLA — Qwen3.6-35B-A3B, rebase base mới + test production

## Bối cảnh

Prod `misa-qwen36-35b-a3b-fp8-api` sập lặp lại: `TypeError: 'NoneType' ... set_mamba_track_indices_from_reqs`
(EAGLE verify + mamba extra_buffer, kích khi ≥2 request chồng nhau). Là bug sglang
(issue #31103 — đúng model này), fix #27998 merge 16/07; image cũ build 13/07 nên dính.
Việc đã làm: rebase image flashqla lên nightly 20260731 → test ổn định → benchmark → 4 bug tìm ra & xử lý trong quá trình test (#1,#2 fix ở r2/r3; #4 fix ở r5; #3 chỉ có mitigation).

## Môi trường đo

- Hackathon H200 (1 GPU/server, không MIG), model `Qwen3.6-35B-A3B` **bf16** (67GB), TP1.
- Prod chạy FP8 + 70% GPU ⇒ **đọc số theo chênh lệch giữa hai arm, không lấy giá trị tuyệt đối làm mốc prod.**
- Image: `ductransa01/sglang-flashqla:cu12-20260731-r3` (sglang `dev1+ga14971730.d20260731` + patch flashqla + 2 fix dưới).
- Cờ chung mọi arm: `mem-fraction-static 0.8 · chunked-prefill-size 16384 · EAGLE 3/1/4 · enable-metrics`.
- Bench: `sglang.bench_serving`, dataset random, output 256, 8 prompt, concurrency 4, `flush_cache` giữa các điểm đo, warmup 32K trước khi đo (đốt JIT TileLang).

## Bug #1-#3 tìm được ở giai đoạn đầu (#4 xem mục CCR A/B)

| # | bug | hậu quả nếu lên prod | fix |
|---|---|---|---|
| 1 | `pip install flash-qla` hạ cấp `apache-tvm-ffi` 0.1.11→0.1.9 (flash-qla ghim cứng dep; base mới cần 0.1.11) | server chết NGAY lúc capture CUDA graph: `TypeError: make_kwargs_wrapper() ... map_dataclass_to_tuple` | cài `--no-deps` + assert version trước/sau (Dockerfile.cu12-rebase). Tag hỏng `cu12-20260731` — ĐỪNG dùng |
| 2 | FlashQLA extend trả `h=None` trong khi mamba radix cache (extra_buffer) cần mid-chunk state khi ranh giới track nằm trong chunk | SIGQUIT ở **forward ≥16384 token đầu tiên** (`assert h is not None`). Prod cũ nhiều khả năng hiếm khi chạm (prompt ~2K, nhưng burst nhiều request VẪN có thể gói đủ 16384/forward — chưa xác nhận từ log prod, đừng dùng làm bằng chứng lịch sử) | r3: call-site truyền `track_needs_h` xuống kernel → fallback Triton cho forward đó (Dockerfile.r3-trackfix). Đã tái hiện crash trên r2 và xác nhận r3 sống 8/8 request 32K, có log fallback |
| 3 | TileLang JIT compile theo shape mới NGAY LÚC PHỤC VỤ, ~18s/lần, chặn cả scheduler | p99 TTFT nhảy ~18s mỗi khi gặp shape đóng gói mới (quan sát 3 lần trong một sweep) | giảm nhẹ: mount `TILELANG_CACHE_DIR` lên PVC (đã có trong config) + pre-warm sau deploy. KHÔNG có fix triệt để |

## Kết quả benchmark

### Arm 1 — cấu hình prod (`auto` = extra_buffer, radix cache bật), backend triton

| input | Mean TTFT | Median TTFT | P99 TTFT | Mean TPOT | tok/s out |
|---|---|---|---|---|---|
| 2048 | 166 ms | 137 | 278 | 3.69 ms | 848 |
| 8192 | 407 ms | 423 | 663 | 4.39 ms | 629 |
| 32768 | 1464 ms | 1343 | 2766 | 9.23 ms | 261 |
| 65536 | 3507 ms | 3407 | 6083 | 16.21 ms | 133 |

Arm flashqla + extra_buffer trên **r2**: không có số — **sập** ở điểm 32768 (bug #2).
Trên **r3** (fallback): 32K cho 8/8 OK, TTFT ≈ triton (1889 vs 1464 ms, chênh gồm lần dispatch đầu) —
tức dưới extra_buffer, flashqla gần như luôn nhường Triton ⇒ **không có lợi ích hiệu năng, chỉ còn an toàn**.

### Arm 2 — so kernel công bằng (`no_buffer` + no-overlap; nền duy nhất flashqla chạy thật trên r2)

| input | triton TTFT mean/med | flashqla TTFT mean/med | triton TPOT | flashqla TPOT |
|---|---|---|---|---|
| 2048 | 251 / 70 | 260 / 72 | 4.31 | 4.27 |
| 8192 | 350 / 264 | **6840 / 263** ⚠️ | 4.86 | 13.30 ⚠️ |
| 32768 | 1294 / 1025 | 1293 / 1026 | 10.02 | 9.84 |
| 65536 | 3117 / 3041 | **2874 / 2225** | 17.99 | 18.88 |

- 2048: y hệt nhau — flashqla không dispatch (ngưỡng 16384 token/forward).
- 8192 ⚠️: mean/P99 của flashqla bị phá bởi **3 khoảng lặng ~18s** = JIT TileLang shape mới (bug #3); median 263ms vẫn ngang triton — không phải kernel chậm, là chi phí compile lần đầu.
- 32768: hoà.
- 65536: flashqla thắng **~8% mean, ~27% median TTFT**; TPOT 18.88 vs 17.99 (+4.9%, trong nhiễu n=8; kernel prefill-only, decode vẫn triton).

### Đối chiếu ghi nhận cũ (13/07, kernel-level 2.7× ở 128K): nhất quán — GDN chỉ ~10% compute prefill
của model MoE này nên 2-3× kernel chỉ ra 4-8% E2E, và chỉ từ 64K trở lên.

## Test ổn định (smoke A/B chạy trên image r2; crash-repro trên r3)

| test | cấu hình | kết quả |
|---|---|---|
| smoke 4 phút × 4 luồng, output lệch 8↔400 tok | extra_buffer + overlap + EAGLE (đúng cấu hình đã sập prod) | **1806 req / 0 lỗi / 0 restart**; prefix cache tái dùng 2048/2113 tok |
| smoke cùng bài | no_buffer + no-overlap | 1736 req / 0 lỗi; **prefix cache = 0 tuyệt đối** ⇒ đừng dùng no_buffer cho prod |
| crash-repro 32K × 8 prompt | flashqla + extra_buffer (r2 chết ngay forward đầu) | r3: **8/8 OK, 0 crash**, log fallback xuất hiện đúng |
| soak 8 phút × 6 luồng, prompt trộn 2K/8K-shared/20K + **client abort giữa stream** (15%) | triton + extra_buffer + overlap + EAGLE | **2332 OK + 412 abort / 0 crash / 0 restart**, health xanh sau soak; 681 request 18-21K (đa chunk 16384) qua sạch. Đây là phép thử gián tiếp mạnh nhất cho kịch bản #28413/#29449 (req bị free khi verify còn giữ) |
| so output greedy 2 arm | cùng prompt ~19.6K tok, temperature 0 | Trùng 86 ký tự đầu rồi rẽ nhánh diễn đạt ("a massive repetition of" vs "a massively repeated"), cả hai mạch lạc, cùng nghĩa, độ dài xấp xỉ (917 vs 865 ký tự). Khớp kỳ vọng: lệch số học bf16 ~5e-4 lật token ở vị trí xác suất sát nhau; parity kernel-level đã pass trước đó (cosine 0.99999) |

## Audit port vs upstream (đối chiếu source FlashQLA 0.1.2, 31/07)

Đối chiếu wrapper `gdn_flashqla.py` với `flash_qla/ops/gated_delta_rule/chunk/__init__.py` upstream:

| hạng mục | wrapper mình | upstream 0.1.2 | kết luận |
|---|---|---|---|
| kwargs truyền vào | `output_final_state, use_qk_l2norm_in_kernel, cu_seqlens, state_v_first` | đều tồn tại trong chữ ký `chunk_gated_delta_rule` (README không liệt kê đủ, source có) | ✅ |
| `scale` không truyền | — | default `1/sqrt(K)`, cùng quy ước FLA/triton fork của sglang | ✅ |
| `state_v_first=True` | pool sglang layout `[N,HV,V,K]` | docstring: mặc định `[N,HV,K,V]`, cờ này đảo → khớp pool | ✅ |
| l2norm trong kernel | bật | upstream áp `l2norm_fwd(q/k)` trước fused kernel — cùng chỗ FLA fork làm | ✅ |
| chunk size | — | `CHUNK_SIZE=64` trên sm90 (=FLA) | ✅ |
| **mid-chunk states `h`** | **dùng high-level API → không lấy được** | **`chunk_gated_delta_rule_fwd(output_h=True)` CÓ trả h** (dùng cho backward) | ⚠️ gốc rễ bug #2 |

⇒ Port dùng API đúng và đủ cho đường suy luận thuần; thiếu sót duy nhất là chọn high-level API nên
không lấy được `h` cho mamba radix-track — mà upstream **có** đường lấy (`output_h=True` ở tầng fwd).

**Đường r4 (chưa làm, nếu muốn flashqla ăn thật dưới extra_buffer):** gọi `chunk_gated_delta_rule_fwd(output_h=True)`
thay high-level. Ba việc phải giải: (1) tự áp `l2norm_fwd` cho q/k (tầng fwd không có cờ đó);
(2) `auto_cp` remap `cu_seqlens` khi cắt context-parallel — `h` trả về ở layout CP-split, phải map ngược
về layout varlen gốc trước khi đưa cho `_track_mamba_state_extend`; (3) khớp granularity: h của flashqla
per-chunk-64, sglang cần per `mamba_cache_chunk_size` — phải subsample đúng index. Việc (2) là chỗ dễ sai
âm thầm nhất ⇒ bắt buộc parity test h-vs-Triton trước khi ship. r3 fallback vẫn là bản ship đúng hôm nay.

Ghi chú định vị của chính Qwen (README): lợi ích rõ nhất ở "**pretraining** scenarios và **edge-side**
agentic inference" — tức fwd+bwd training và inference model dense nhỏ. Khớp số đo của mình: serving
MoE 35B (GDN ~10% compute prefill) chỉ ra 0-8% E2E, và chỉ từ 64K.

## r4 — native h: flashqla chạy thật CÙNG radix cache (làm theo yêu cầu CCR/long-context)

Đường r4 nêu ở audit đã làm xong trong cùng ngày, vì probe cho thấy không cần map gì:

**Probe parity (H200, GDN Qwen3.6, varlen 2 seq 20480+12288):** `chunk_gated_delta_rule_fwd(output_h=True)`
của flash_qla trả `h` **trùng fork sglang cả shape (1,512,32,128,128), granularity 64 tok/chunk, hướng K/V
lẫn giá trị** (mean 2.2e-5; layout transposed sai gấp 250× ⇒ loại nhầm lẫn). `auto_cp` map về layout varlen
gốc ngay trong thư viện. `o` parity 1.7e-3, `final_state` 5e-4.

**r4** (`Dockerfile.r4-native-h` + `flashqla_native_h_r4.py`, chồng lên r3): khi call-site báo `track_needs_h`,
wrapper tự `l2norm_fwd(q/k)` rồi gọi API tầng thấp `output_h=True`, trả `h` thẳng cho
`_track_mamba_state_extend` — **hết fallback, giữ nguyên prefix cache**.

**Kiểm r4 (image `cu12-20260731-r4`):**

| test | kết quả |
|---|---|
| T4 state-reuse: prompt 20.3K → track h → prompt chung prefix (cache hit 20288/20333) → so với fresh-recompute cùng arm | Lệch cached-vs-fresh **nhỏ hơn cả arm Triton đối chứng** (prefix chung 371 vs 201 ký tự; Triton thuần cũng lệch ⇒ nguồn lệch là bản chất đường cached-continuation, không phải h). Cả hai output mạch lạc ⇒ **h đúng** |
| bench + extra_buffer, JIT ấm | 32K: 1467/1336 ms — hoà triton (1464/1343) · **65K: 3305/3211 ms = −6% TTFT** so triton (3507/3407) · TPOT không đổi |
| bench lần ĐẦU (JIT nguội) | 32K mean 10080ms, P99 ~20s — biến thể kernel `output_h=True` compile ngay trong lúc phục vụ. **Bằng chứng bug #3 áp cả đường native-h ⇒ pre-warm sau deploy là BẮT BUỘC với config flash** |
| soak 3 phút native-h + abort (15%) | **589 OK + 110 abort / 0 crash**, 156 request 18-21K qua đường native-h, p95 1.85s |
| boot `r4-shim` với `RUNAI_PATH_PREFIX` như prod | bare `/health` 200 (probe RunAI) · prefixed health/chat/metrics 200 (ingress) · bare `/metrics` 200 (scraper nội bộ) — **5/5**; middleware `_StripPrefix` gắn đúng, `sitecustomize.py` md5 khớp bản prod cũ |
| bench trên ĐÚNG bản ship `r4-shim` (middleware bật) | 2K: 374/385 · 32K: 1731/1422 · 65K: 3493/3357, TPOT 15.81. Chênh vs r4 trần nằm trong nhiễu run-to-run (n=8) + cold-start server mới (mean lệch nhiều hơn median, chênh CO LẠI theo kích thước, TPOT không đổi) ⇒ **shim không gây regression hệ thống** — đúng kỳ vọng: 1 phép `startswith`/request |

⇒ Với CCR: lượt đầu conversation / cache-miss với context ≥32K đi qua flashqla mà **không mất prefix cache**;
lợi ích tăng theo độ dài (65K −6%, kernel-level 128K 2.7× ⇒ kỳ vọng −10-15% ở 128K, chưa đo E2E).
Dưới 16K/forward vẫn Triton — đó là crossover vật lý đã đo (flashqla thua dưới ngưỡng do CP preprocessing),
không phải giới hạn tích hợp.

## CCR A/B dưới tải (làm theo yêu cầu "so với tải + context length")

Tool: `flashqla-loadtest/bench_ccr.py` — chính generator đội đã dùng bài bloom, dựng từ profile
CCR monitoring thật (bucket 10-20K nặng 553 · 20-30K nặng 458 · 70-80K nặng 150 ⇒ avg input ~26.7K,
output 150, ignore_eos). Hai arm cùng image, cùng seed, chạy song song 2 GPU, khác đúng cờ backend.
Mức tải 8 & 24 concurrent (mức 50-1160 trong tool là cho cả cụm gateway; đây 1 GPU/arm).

### Phát hiện thêm bug #4 nhờ vòng 1 (r4), đã fix thành r5

r4 gọi API tầng thấp với `enable_fwd_cp_cache` mặc định **False** (tầng cao dùng True) ⇒
**COLD ccr=8 TTFT p95 +205%, throughput −15%**; r5 = một dòng `enable_fwd_cp_cache=True`, số về sạch.
**Cơ chế (đã sửa sau audit độc lập đọc source flash_qla 0.1.2):** cờ này KHÔNG phải cache cross-forward —
nó chọn biến thể kernel `get_warmup_chunks` (single) vs `get_warmup_chunks_bidi` (bidi); đường high-level
luôn dùng bidi nên biến thể bidi đã được JIT-warm sẵn, còn single (đường r4) là kernel TileLang CHƯA warm
⇒ giả thuyết hợp lý nhất cho +205% là first-touch của biến thể đó (bộ đếm "begins to compile"=0 của phép
đo đã bỏ sót lớp JIT/load này). Số đo trước/sau là thật; lời giải cơ chế ở mức giả thuyết-khớp-nhất,
CHƯA cô lập tuyệt đối — nếu regression dạng này tái xuất, soi TILELANG_CACHE_DIR trước tiên.
Kèm theo từ audit: `tensor_cache` của flash_qla key theo identity ⇒ KHÔNG có rủi ro cache trả nhầm giữa
các forward. Đo lại cùng điều kiện (2 arm boot mới, song song):

| COLD (cache miss) | ccr=8 r4→r5 | ccr=24 r4→r5 |
|---|---|---|
| TTFT p95 vs triton | +205% → **+0.8%** | +42% → **−5.4%** |
| TTFT p99 vs triton | +198% → **+0.6%** | +29% → **−29.2%** (thắng ngược) |
| TTFT p50 vs triton | — → +16.7% | — → +13.0% |
| out tok/s vs triton | −15% → **+3.6%** | ≈ → +1.5% |

### Bảng chốt CCR (r5 cold + r4 warm — warm hầu như không đi qua đường native-h nên vẫn giá trị)

| chế độ | ccr | TTFT p50 | TTFT p95 | TTFT p99 | TPOT | tok/s |
|---|---|---|---|---|---|---|
| WARM (90% cache — CCR ổn định) | 8 | +9% | +3% | +5% | −2% | +1.3% |
| WARM | **24** | −1% | **−16%** | **−18%** | ≈ | +1.3% |
| COLD (lượt đầu / cache miss) | 8 | +17% | +0.8% | +0.6% | −9% | +3.6% |
| COLD | **24** | +13% | −5% | **−29%** | +5% | +1.5% |

(dấu − = flashqla tốt hơn; n=170-800/ô; err=0 toàn bộ; bf16 H200 — đọc theo chênh lệch, không lấy tuyệt đối)

**Kết luận hơn thua cho CCR:** ở tải cao (ccr 24) flashqla **thắng đuôi TTFT 16-29%** cả warm lẫn cold —
đúng trục SLA của CCR; median cold chịu +13-17% (chi phí cố định đường native-h); throughput nhỉnh 1-4%;
TPOT hoà. Tải thấp hai arm tương đương. Không lỗi, không crash ở mọi ô đo.

### CCU sweep 24→200 (format bảng bloom 22/07: cold + temp 0, r5-shim, thường → flashqla)

| CCU | Error Rate | TTFT p95 | E2E p50 | Eff.Conc (req/s × E2E p50) |
|---|---|---|---|---|
| **24** | 0% → 0% | **10.4s → 7.2s (−30%)** | **17.0 → 15.5s (−9%)** | 23 → 22 |
| 48 | 0% → 0% | 15.4 → 16.3s (+6%) | 32.8 → 31.8s (−3%) | 41 → 40 |
| 80 | 0% → 0% | 40.2 → 37.8s (−6%) | 54.5 → 53.4s (−2%) | 60 → 60 |
| 96 | 0% → 0% | 51.3 → 50.3s (−2%) | 64.9 → 64.8s (0%) | 67 → 68 |
| 128 | 0% → 0% | 71.7 → 71.4s (0%) | 86.2 → 85.8s (−1%) | 78 → 80 |
| 200 | 0% → 0% | 119.3 → 118.6s (−1%) | 133.7 → 131.8s (−1%) | 99 → 101 |

Đọc: kernel ăn **−30% TTFT p95 / −9% E2E ở vùng chưa bão hoà** (CCU 24 ≈ capacity 1 GPU với avg input
26.7K); từ CCU 48 hệ queue-bound nên hai arm hội tụ (xếp hàng nuốt lợi ích kernel — vật lý, không phải
kernel yếu). TPOT hoà mọi mức; err=0 toàn dải (sglang xếp hàng, không timeout — bảng bloom cũ có error
vì gateway nginx cắt). ⚠️ So với bảng bloom (−21…−27% đều mọi CCU): bài đó gộp cả nâng cấp base sglang
lẫn kernel + méo do error truncation; bài này cô lập đúng MỘT biến (cờ backend) — r5 ship mang cả hai
nên tổng lợi ích so prod cũ ≥ bảng bloom ở vùng vận hành.

⚠️ Ghi chú trung thực về biên đo: hai phép đo cold ~24 concurrent cho chênh lệch KHÁC nhau — CCR A/B round2
p95 −5.4% vs CCU sweep −30% (khác thứ tự level, độ ấm server, n≈175/ô). Khoảng tin cậy thực tế cho
"lợi ích đuôi TTFT vùng chưa bão hoà" nên đọc là **−5%…−30%**, đừng trích mỗi số đẹp. Tương tự, bảng
"Bảng chốt CCR" trộn WARM đo trên r4 với COLD đo trên r5 — warm hầu như không đi qua native-h nên vẫn
dùng được, nhưng về phương pháp thì chưa đồng nhất image (chiều lệch nếu có là CÓ LỢI cho triton).

## Audit độc lập (Fable, 31/07) — kết quả xử lý

Một agent audit độc lập soi toàn bộ chuỗi r2→r5-shim + tài liệu. Findings và trạng thái:

| finding | severity | xử lý |
|---|---|---|
| `deployment_name` config flash TRÙNG config chính (lỗi có sẵn từ config gốc) → deploy đè workload chính + prefix lệch → 404 | **blocker** | ✅ sửa thành `...-flash-api` |
| Runbook/Khuyến nghị còn trỏ r3/r4-shim (bản bị chính báo cáo cấm) | **blocker** | ✅ về r5-shim nhất quán |
| `--no-deps flash-qla` không pin version → rebuild sau trôi | major | ✅ pin `==0.1.2` |
| Giải thích cơ chế bug #4 sai (không phải cache cross-forward; là biến thể kernel warmup single/bidi) | major | ✅ sửa lời giải, đánh dấu mức giả thuyết |
| Cast state fp32→bf16 chưa được ghi nhận là accepted risk + thiếu canary | major | ✅ mục Accepted risks + canary accept_len |
| `SGLANG_ROOT_PATH` là env chết + nguy cơ strip prefix 2 lần nếu run_server.sh dùng cả 2 cơ chế | major | ✅ ghi vào runbook bước 4 |
| Điều kiện `track_needs_h` (r3) có thể hẹp hơn consumer → còn corner crash | major (PLAUSIBLE) | ✅ **đã xác minh trên IMAGE** (docker exec): `assert h`/`squeeze` nằm TRONG guard `numel()>0` — khớp điều kiện r3, an toàn; cảnh báo đến từ checkout local lệch image |
| Số mâu thuẫn giữa các bảng (−5.4% vs −30%; "TPOT không đổi" +4.9%; bảng trộn r4/r5) | major | ✅ sửa wording + footnote biên đo −5%…−30% |
| `query_start_loc.to(long)` không copy — bẫy nếu upstream đổi dtype buffer | minor | ✅ thêm `copy=True` vào script cho build sau (r5 đã ship vẫn an toàn trên base này — int32 luôn copy) |
| Prewarm coverage là heuristic, chưa tự kiểm chứng; shim thiếu boundary check `/`; API key plaintext trong YAML | minor | ghi nhận — không chặn ship; xử lý đợt sau |

## Accepted risks (ghi rõ, có canary — bổ sung sau audit độc lập)

1. **Cast state fp32→bf16 ở đường flashqla** (cả high-level lẫn native-h): Triton truyền pool fp32 trực
   tiếp; flashqla gather `ssm_states[idx].to(bf16)` + ghi lại `final_state`/`h` bf16 → mỗi ranh giới
   forward ≥16K và mỗi checkpoint radix-cache bị round ~8 bit mantissa; decode + EAGLE verify khởi động
   từ state đã round. T4/parity chỉ chứng minh "không hỏng thô", CHƯA đo chất lượng sinh. **Canary:
   `accept_len` trong log prod — tụt dần không kèm lỗi = dấu hiệu state degradation, báo lại ngay.**
2. **Tổ hợp `tilelang 0.1.11 + flash-qla 0.1.2` upstream CHƯA test** (flash-qla ghim 0.1.9, mình --no-deps
   để giữ 0.1.11 cho base): đã validate thực nghiệm (parity + toàn bộ bench) nhưng là tổ hợp riêng của mình.
   Đã pin `flash-qla==0.1.2` trong Dockerfile để rebuild không trôi version.
3. **#29449 (upstream, open)**: guard None→0 có thể ghi state req đã freed vào slot tái cấp phát — hỏng
   âm thầm. Soak abort-heavy là phép thử gián tiếp; không có oracle tuyệt đối.
4. **Bench bf16/H200-nguyên vs prod FP8/70%GPU**: mọi % là chênh lệch giữa hai arm cùng điều kiện,
   không phải SLA tuyệt đối cho prod.

## Khuyến nghị

1. **Config chính (`misa-qwen36-35b-a3b-fp8-api`)**: lên image **r5-shim** (một image dùng chung — không
   truyền cờ thì mặc định triton, y hệt sglang thường), **bỏ cờ `linear_attn_prefill_backend`**.
   Workload thật ~2K token/prompt: flashqla không dispatch ⇒ 0 lợi ích, còn gánh rủi ro JIT.
   Giữ `mamba_radix_cache_strategy` mặc định (`auto`→extra_buffer) — fix #27998 đã kiểm dưới tải chồng request.
2. **Config flash (`...-flash-api`) — cũng r5-shim**, giữ cờ flashqla: ăn thật ở context dài mà vẫn giữ
   prefix cache (65K −6% TTFT; đuôi CCR −16…−29% tải cao; state-reuse đã kiểm đúng). Cho traffic CCR/agentic.
   ⛔ KHÔNG dùng r4-shim (bug #4: cold TTFT p95 +205%) hay r3 (fallback liên tục, lợi ích ≈ 0).
3. **Sau deploy config flash: pre-warm là BẮT BUỘC**, không phải khuyến nghị suông — bắn vài request ≥32K
   (kèm 1 lượt có shared-prefix để kích cả đường `output_h=True`) cho đủ tổ hợp shape; không làm thì
   những request dài đầu tiên chịu stall JIT ~18s/shape (đo được, lặp lại). Cache JIT nằm trên PVC
   (`TILELANG_CACHE_DIR`) nên chỉ cần làm một lần cho mỗi image mới.
4. Rủi ro còn treo (upstream, chưa fix): #29449 — guard None→0 có thể ghi state req đã freed vào slot
   tái cấp phát (hỏng âm thầm, không crash). Soak abort-heavy (T2) là phép thử gián tiếp; không có oracle
   thì không phát hiện tuyệt đối được. Theo dõi PR upstream.

## Runbook lên PROD (các bước còn lại — cần mạng nội bộ, máy dev này không với tới)

```bash
# 1. Keo image (da push san Docker Hub, KHONG can build gi them)
docker pull ductransa01/sglang-flashqla:cu12-20260731-r5-shim   # digest f5cf60cf2d8f

# 2. Retag + day Harbor RDP (tag versioned — aiteam PROD bat Tag Immutability)
docker tag  ductransa01/sglang-flashqla:cu12-20260731-r5-shim \
            <HARBOR>/aiteam/sglang-flashqla:cu12-20260731-r5-shim
docker push <HARBOR>/aiteam/sglang-flashqla:cu12-20260731-r5-shim

# 3. Jenkins AITeam/.push-image -> TEST <HARBOR-TEST> -> PROD <HARBOR-PROD>

# 4. Deploy 2 workload (thu tu: config CHINH truoc — no dang crash):
#    - misa-qwen36-35b-a3b-fp8-api.yaml        (da tro r5-shim; NHO ra soat
#      /home/local/data/run_server.sh tren PVC — co chinh la noi giu co that,
#      doi/bo --mamba-scheduler-strategy neu co; xac nhan chi MOT co che prefix:
#      RUNAI_PATH_PREFIX (shim) HOAC --fastapi-root-path, KHONG ca hai — hai cai
#      cung bat se strip prefix HAI LAN -> 404; env SGLANG_ROOT_PATH la env chet,
#      sglang khong doc, xoa duoc)
#    - misa-qwen36-35b-a3b-fp8-flash-api.yaml  (da tro r5-shim, flashqla + EAGLE;
#      AUDIT B1: deployment_name da sua thanh ...-flash-api — truoc do trung ten
#      config chinh, deploy se de workload chinh + prefix lech -> 404)

# 5. PRE-WARM config flash (BAT BUOC, mot lan cho moi image moi):
python3 sglang-flashqla/prewarm_flashqla.py \
  --base-url https://<GATEWAY-HOST>/prod-llm/misa-qwen36-35b-a3b-fp8-flash-api \
  --api-key  sk-proj-runai-... \
  --model    misa-qwen3.6-35b-a3b-fp8
# chay 2 lan: lan 1 cham (JIT dang dot), lan 2 moi vong <15s = cache PVC da an

# 6. Nghiem thu: theo doi log ~30 phut gio cao diem, tim
#    "Scheduler hit an exception" (phai = 0) va accept len (phai ~2.5-3.5).
#    accept_len con la CANARY cho rui ro state-cast bf16 (xem muc Accepted risks):
#    accept_len tut dan ma khong co loi = dau hieu state hong am tham -> bao lai

# 7. THU HOI PAT dckr_pat_nHisd... (da di qua hoi thoai + argv shell may chung)
```

Shim đã kiểm 2 tầng trước khi push: middleware `_StripPrefix` gắn đúng + strip prefix
và bare path đều 200 (ASGI test), và boot server thật với `RUNAI_PATH_PREFIX` (xem bảng test).

## Tags

- **`cu12-20260731-r5-shim`** (digest `f5cf60cf2d8f...`) — **bản lên PROD**, dùng chung cho cả hai config
- `cu12-20260731-r5` (digest `cddf85bb7762...`) — như trên, chưa lớp shim (= r4 + enable_fwd_cp_cache=True)
- `cu12-20260731-r4-shim` / `-r4` — ⚠️ dính bug #4: CCR cold TTFT p95 +205%, đừng ship
- `cu12-20260731-r3` — bảo thủ: flashqla fallback thay vì native-h (an toàn, lợi ích ≈ 0)
- `cu12-20260731-r2` — thiếu trackfix: flashqla+extra_buffer sập ở prefill ≥16K (triton thì an toàn)
- `cu12-20260731` — ⛔ hỏng (bug #1), không dùng
- Config: `MISA.ModelConfig/workloads/qwen/misa-qwen36-35b-a3b-fp8-api.yaml` (chính)
  + `misa-qwen36-35b-a3b-fp8-flash-api.yaml` (flash) — cả hai đã trỏ r5-shim
- Pre-warm: `sglang-flashqla/prewarm_flashqla.py`
