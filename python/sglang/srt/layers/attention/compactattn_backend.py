"""CompactAttention prefill backend (arXiv 2605.16839) dung nen FlashAttention v4.

Y tuong cua bai bao: dung coi mat na block-sparse 2D nhu MOT TIN HIEU CHON KV, gom cac
block duoc chon lai thanh mot bang KV lien tuc, roi chay kernel FlashAttention DAY tren
phan da gom. Doi mot it cong thua lay do chiem dung SM cao hon nhieu so voi kernel sparse
that su.

Vi sao ban nay khong can viet CUDA:

  * `flash_attn_with_kvcache` cua FA4 nhan `page_table` (flash_attention_v4.py:222).
  * Ta chay `page_size = 1`, nen `page_table` la danh sach chi so THEO TUNG TOKEN.
    Muon model nhin tap token nao thi dung bang dung tap do.
  * Ngu nghia nhan qua duoc giu nguyen mien phi: trong duong extend, FA coi q_len hang
    CUOI cua bang la vi tri cua cac token truy van. Nen neu ta nen phan tien to va giu
    nguyen phan token moi o cuoi, thi truy van van nhin duoc toan bo tien to da chon
    CONG tam giac nhan qua cua chinh chunk -- dung mot lan goi kernel, khong can
    merge_state.

Vi sao Gemma4 bat buoc phai di duong nay: 5 lop full_attention cua no la 16 head truy
van x 512 va 2 KV head x 512 (do thang tren GPU). FlashPrefill V2 la ban fork cua FA3
nen chan head_dim o 256 va tu choi thang. FA4 chay duoc 512.

Chi ap cho lop full_attention. 25 lop sliding-window cua so 1024 da thua san, gom them
khong duoc gi, uy quyen nguyen ven cho backend day.

Giai doan hien tai: `--compactattn-select all` (mac dinh) chon TOAN BO tien to, tuc
ket qua phai TRUNG KHIT voi fa4 thuan. Do la cua kiem tinh dung dan cua phan ong dan
truoc khi cam bo chon vao.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner


class CompactAttnBackend(FlashAttentionBackend):
    """Gom KV theo block roi chay FA4 day tren phan da gom.

    KE THUA thay vi boc ngoai. Ban boc ngoai da that bai ba lan vi hop dong giua
    attention backend voi KV pool va cuda-graph runner rong va ngam: metadata bi dat
    len lop boc con backend day ben duoi van None, roi no o metadata.swa_out_cache_loc;
    va dung lai metadata trong forward_decode thi pha cuda graph
    (CUBLAS_STATUS_EXECUTION_FAILED luc capture). Ke thua thi chi co MOT doi tuong,
    MOT ban metadata, moi duong khac chay nguyen ven cua lop cha.
    """

    def __init__(self, model_runner: "ModelRunner"):
        # FA4 la bat buoc, khong phai lua chon: head_dim 512 cua lop full Gemma4 vuot
        # tran 256 cua v3.
        super().__init__(model_runner, fa_impl_ver=4)

        sa = model_runner.server_args
        self.select_mode = sa.compactattn_select
        self.block_n = sa.compactattn_block_n
        self.topk_blocks = sa.compactattn_topk_blocks
        self.sink_blocks = sa.compactattn_sink_blocks
        self.local_blocks = sa.compactattn_local_blocks
        self.min_prefix_len = sa.compactattn_min_prefix_len
        self.compact_context_len = model_runner.model_config.context_len
        self.compact_page_size = model_runner.page_size
        self._bang_cache = None

    def init_forward_metadata(self, forward_batch: "ForwardBatch"):
        # Bang gom cua che do 'all'/'strided' duoc dung lai giua 5 lop full trong
        # CUNG mot lot. Xoa o dau moi lot de khong bao gio doc phai bang cua lot truoc
        # (id(forward_batch) co the bi cap phat lai sau GC).
        self._bang_cache = None
        return super().init_forward_metadata(forward_batch)

    # ------------------------------------------------------------------ #
    # Extend: chi doi bang KV roi goi lai duong day cua lop cha            #
    # ------------------------------------------------------------------ #
    def _is_swa_layer(self, layer: "RadixAttention") -> bool:
        sw = layer.sliding_window_size
        return sw is not None and 0 < sw < self.compact_context_len

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if self._is_swa_layer(layer):
            return super().forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
            )

        md = self.forward_metadata
        table = (
            None
            if md is None
            else self._compact_page_table(md, forward_batch, layer, q)
        )
        if table is None:
            return super().forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
            )

        page_table, cache_seqlens = table
        saved_table = md.page_table
        saved_seqlens = md.cache_seqlens_int32
        md.page_table = page_table
        md.cache_seqlens_int32 = cache_seqlens
        try:
            return super().forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
            )
        finally:
            md.page_table = saved_table
            md.cache_seqlens_int32 = saved_seqlens

    # ------------------------------------------------------------------ #
    # Dung bang KV da nen                                                  #
    # ------------------------------------------------------------------ #
    def _compact_page_table(self, md, forward_batch: "ForwardBatch", layer, q):
        """Tra ve (page_table_nen, cache_seqlens_nen), hoac None de di duong day.

        Bang gom: [tien to da chon] ++ [toan bo token moi], can trai. Phan token moi
        nam CUOI nen ngu nghia nhan qua cua FA giu nguyen.

        TOAN BO phan chon va dung bang chay bang thao tac tensor. Ban dau phan nay
        dung list Python (range roi extend roi as_tensor)
        -- moi lop moi lot phai dung mot list vai nghin phan tu roi moi chuyen sang
        tensor. Tren pod chi xin 0.1 core CPU thi do la chi phi that, va no chi lo ra
        o che do multi-turn co prefix cache (tien to lon ngay tu luot hai) chu khong
        lo ra khi do TTFT cua request moi tinh.
        """
        if self.compact_page_size != 1:
            return None
        seq_lens = forward_batch.seq_lens_cpu
        ext_lens = forward_batch.extend_seq_lens_cpu
        if seq_lens is None or ext_lens is None or len(seq_lens) != len(ext_lens):
            return None

        seq_lens = [int(x) for x in seq_lens]
        ext_lens = [int(x) for x in ext_lens]
        prefix_lens = [s - e for s, e in zip(seq_lens, ext_lens)]
        if max(prefix_lens, default=0) < self.min_prefix_len:
            # Duoi nguong thi chi phi dung chi muc lon hon phan tiet kiem.
            return None

        # 'all' va 'strided' khong phu thuoc noi dung lop nao, nen bang giong het
        # nhau cho ca 5 lop full. Dung mot lan roi dung lai trong cung mot lot.
        khoa = None
        if self.select_mode != "score":
            khoa = (id(forward_batch), md.page_table.data_ptr(), tuple(seq_lens),
                    tuple(ext_lens))
            if self._bang_cache is not None and self._bang_cache[0] == khoa:
                return self._bang_cache[1]

        diem = None
        if self.select_mode == "score":
            diem = self._cham_diem_block(md, layer, q, prefix_lens, ext_lens)

        table = md.page_table
        device = table.device
        giu = [
            self._chi_so_tien_to(plen, device, None if diem is None else diem[i])
            for i, plen in enumerate(prefix_lens)
        ]
        new_len = [int(g.numel()) + e for g, e in zip(giu, ext_lens)]
        width = max(new_len)
        out = torch.zeros((len(seq_lens), width), dtype=table.dtype, device=device)
        for i, (idx, plen, elen) in enumerate(zip(giu, prefix_lens, ext_lens)):
            n = int(idx.numel())
            if n:
                out[i, :n] = table[i].index_select(0, idx)
            if elen:
                out[i, n : n + elen] = table[i, plen : plen + elen]
        seqlens = torch.tensor(new_len, dtype=torch.int32, device=device)

        ket_qua = (out, seqlens)
        if khoa is not None:
            self._bang_cache = (khoa, ket_qua)
        return ket_qua

    def _cham_diem_block(self, md, layer, q, prefix_lens, ext_lens):
        """Diem cho tung block KV cua tien to, tung request.

        Kieu FlashPrefill/Quest, training-free: gop trung binh K trong moi block roi
        cham voi Q gop trung binh cua chunk hien tai. Hop nhat theo KV head bang max
        (`page_table` la theo REQUEST chu khong theo head, nen bang gom phai dung
        chung cho ca hai head -- lay max de mot block quan trong voi bat ky head nao
        cung duoc giu).

        Tra ve list[Tensor] diem theo block, hoac None neu khong lay duoc K cache.
        """
        try:
            k_buf = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        except Exception:
            return None
        if k_buf is None:
            return None

        bn = self.block_n
        table = md.page_table
        # Q gop trung binh cua chunk, quy ve tung KV head (GQA)
        # q: [tong_token, q_head, head_dim]
        nq = q.shape[1]
        nkv = k_buf.shape[1]
        nhom = max(nq // max(nkv, 1), 1)
        ra = []
        bat_dau = 0
        for i, (plen, elen) in enumerate(zip(prefix_lens, ext_lens)):
            if plen <= 0 or elen <= 0:
                ra.append(None)
                bat_dau += elen
                continue
            q_i = q[bat_dau : bat_dau + elen]  # [elen, nq, d]
            bat_dau += elen
            q_m = q_i.mean(dim=0)  # [nq, d]
            q_m = q_m[: nhom * nkv].view(nkv, nhom, -1).mean(dim=1)  # [nkv, d]

            chi_so = table[i, :plen].to(torch.long)
            k = k_buf.index_select(0, chi_so)  # [plen, nkv, d]
            du = (-plen) % bn
            if du:
                k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, du))
            k = k.view(-1, bn, k.shape[1], k.shape[2]).mean(dim=1)  # [nblock, nkv, d]
            diem = torch.einsum("bhd,hd->bh", k.float(), q_m.float())
            ra.append(diem.amax(dim=1))  # hop nhat theo head
        return ra

    def _chi_so_tien_to(self, prefix_len: int, device, diem=None):
        """Tensor chi so token cua tien to duoc giu. Khong co list Python nao.

        `all`      giu het -- cua kiem tinh dung dan, phai trung fa4 thuan.
        `strided`  sink + cua so cuoi + lay thua deu o giua. KHONG doc noi dung K;
                   chi de do tran toc do tach khoi cau hoi chon kheo toi dau.
        `score`    sink + cua so cuoi + top-k theo diem tu `_cham_diem_block`.
        """
        if prefix_len <= 0:
            return torch.empty(0, dtype=torch.long, device=device)
        if self.select_mode == "all":
            return torch.arange(prefix_len, dtype=torch.long, device=device)

        bn = self.block_n
        n_blocks = (prefix_len + bn - 1) // bn
        bat_buoc = torch.cat([
            torch.arange(min(self.sink_blocks, n_blocks), dtype=torch.long, device=device),
            torch.arange(max(0, n_blocks - self.local_blocks), n_blocks,
                         dtype=torch.long, device=device),
        ])
        bat_buoc = torch.unique(bat_buoc)
        ngan_sach = self.topk_blocks - int(bat_buoc.numel())

        chon = bat_buoc
        if ngan_sach > 0 and n_blocks > int(bat_buoc.numel()):
            con_lai = torch.ones(n_blocks, dtype=torch.bool, device=device)
            con_lai[bat_buoc] = False
            if diem is not None and diem.numel() >= n_blocks:
                d = diem[:n_blocks].clone()
                d[~con_lai] = float("-inf")
                k = min(ngan_sach, int(con_lai.sum()))
                them = torch.topk(d, k).indices
            else:
                ids = torch.nonzero(con_lai, as_tuple=True)[0]
                buoc = max(1, int(ids.numel()) // ngan_sach)
                them = ids[::buoc][:ngan_sach]
            chon = torch.cat([bat_buoc, them])

        chon = torch.unique(chon)  # unique tra ve da sap xep
        offs = torch.arange(bn, dtype=torch.long, device=device)
        idx = (chon.unsqueeze(1) * bn + offs.unsqueeze(0)).reshape(-1)
        return idx[idx < prefix_len]
