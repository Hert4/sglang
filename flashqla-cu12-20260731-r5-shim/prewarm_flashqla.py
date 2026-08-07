#!/usr/bin/env python3
"""Pre-warm BẮT BUỘC sau mỗi lần deploy image flashqla mới (config -flash-api).

Vì sao: TileLang JIT compile theo shape NGAY LÚC PHỤC VỤ, ~18s/shape, chặn cả
scheduler (đo 31/07, lặp lại được). Cache nằm ở TILELANG_CACHE_DIR trên PVC nên
mỗi image mới chỉ cần đốt MỘT lần; không đốt thì những request dài đầu tiên của
người dùng thật chịu stall.

Bốn vòng, mỗi vòng nhắm một họ shape:
  R1  1×33K            — chunk 16384 đơn seq + biến thể native-h (output_h=True)
  R2  4×9K đồng thời   — forward 16384 GHÉP nhiều seq (2×8192...)
  R3  1×49K            — số chunk lớn hơn (bảo hiểm rẻ)
  R4  33K + đuôi mới   — đường khôi phục state từ radix cache (không JIT, kiểm nhanh)

Chạy từ máy vào được gateway:
  python3 prewarm_flashqla.py \
    --base-url https://<GATEWAY-HOST>/prod-llm/misa-qwen36-35b-a3b-fp8-flash-api \
    --api-key  sk-proj-runai-... \
    --model    misa-qwen3.6-35b-a3b-fp8

Đọc kết quả: lần đầu các vòng R1/R2 chậm (JIT đang chạy — đúng mục đích);
chạy LẠI script lần 2 mọi vòng phải nhanh (<15s/vòng). Lần 2 chậm = cache PVC
không ăn, kiểm tra mount TILELANG_CACHE_DIR.
"""
import argparse
import concurrent.futures as cf
import json
import ssl
import time
import urllib.request

FILLER = "digital transformation data pipeline observability metric "  # ~7 token


def chat(base, key, model, content, max_tokens, timeout=900):
    req = urllib.request.Request(
        f"{base.rstrip('/')}/v1/chat/completions",
        data=json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"},
    )
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE  # gateway noi bo dung cert noi bo
    # bo qua proxy corp (http_proxy/https_proxy env) — day la mang noi bo,
    # di qua proxy se hong ket noi kho hieu (audit m2)
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=ctx),
    )
    t0 = time.time()
    body = json.loads(opener.open(req, timeout=timeout).read())
    dt = time.time() - t0
    u = body.get("usage", {})
    return dt, u.get("prompt_tokens")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--model", required=True)
    a = ap.parse_args()

    def go(tag, content, max_tok):
        dt, ptok = chat(a.base_url, a.api_key, a.model, content, max_tok)
        print(f"  {tag}: {dt:6.1f}s  prompt_tokens={ptok}")
        return dt

    p33k = FILLER * 4700 + " prewarm-R1"
    print("R1: 1x33K (chunk don seq + native-h)")
    go("33K", p33k, 8)

    print("R2: 4x9K dong thoi (forward 16384 ghep seq)")
    with cf.ThreadPoolExecutor(4) as ex:
        futs = [ex.submit(chat, a.base_url, a.api_key, a.model,
                          FILLER * 1300 + f" prewarm-R2-{i}", 8) for i in range(4)]
        for i, f in enumerate(futs):
            dt, ptok = f.result()
            print(f"  9K#{i}: {dt:6.1f}s  prompt_tokens={ptok}")

    print("R2b: 2x17K dong thoi (multi-seq >=16K, ho shape CP-off — audit m2)")
    with cf.ThreadPoolExecutor(2) as ex:
        futs = [ex.submit(chat, a.base_url, a.api_key, a.model,
                          FILLER * 2450 + f" prewarm-R2b-{i}", 8) for i in range(2)]
        for i, fu in enumerate(futs):
            dt, ptok = fu.result()
            print(f"  17K#{i}: {dt:6.1f}s  prompt_tokens={ptok}")

    print("R3: 1x49K")
    go("49K", FILLER * 7000 + " prewarm-R3", 8)

    print("R4: 33K + duoi moi (khoi phuc state tu cache — phai NHANH)")
    dt = go("33K+Q", p33k + "\n\nTom tat 1 cau.", 32)
    if dt > 30:
        print("  ⚠️ R4 cham bat thuong — cache/track co van de, xem log server")

    print("XONG. Chay lai script lan 2: moi vong phai <15s thi cache PVC da an.")


if __name__ == "__main__":
    main()
