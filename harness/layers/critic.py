"""LỚP `critic` — bài giảng Day 16, §2 (Reflection & Self-Critique).

NHIỆM VỤ: mô hình KHÔNG BAO GIỜ nói "tôi không biết". `abstain` bị gán
cứng `False`, và nó bịa theo ba kiểu khác nhau:

  (a) brief `absent`  -> bịa ra một con số không có trong tài liệu nào.
  (b) không có bằng chứng -> bịa ra một câu chung chung vô thưởng vô phạt.
  (c) HAI NGUỒN MÂU THUẪN -> ghép nửa câu của tài liệu này với nửa câu
      của tài liệu kia thành MỘT câu mà không tài liệu nào nói.

TÍN HIỆU (chỉ một dòng): câu trong `claim["text"]` có xuất hiện NGUYÊN VĂN
trong bằng chứng agent đã thực sự đọc hay không —

    text in ctx.observed_text

Trên một brief có bằng chứng tốt thì mọi claim đều thoả điều kiện này,
nên critic xây trên tín hiệu đó không báo động giả.

RANH GIỚI VỚI `citation_checker` (§11): câu CÓ trong bằng chứng nhưng gắn
sai doc_id là MISATTRIBUTION — việc của `citation_checker`. Câu KHÔNG có
trong bất kỳ bằng chứng nào là FABRICATION — việc của bạn ở đây. Hai điều
kiện loại trừ nhau, đừng làm phần việc của lớp kia.

ĐIỂM SỐ (đọc kỹ, đây là nơi kiếm nhiều điểm nhất):
  * Một claim bịa bị chấm `HALLUCINATED`: mất điểm precision VÀ mất trọn
    15 điểm honesty, trên MỌI brief.
  * Trên brief `is_absent`, `abstain: true` được 0.75 recall + trọn 15
    điểm honesty. "Không có số liệu" CHÍNH LÀ câu trả lời đúng.
  * Trên brief mâu thuẫn, ĐỪNG trông đợi "nêu cả hai phía" tự động cho
    recall đầy đủ: recall chấm THEO TỪNG required_fact bằng key terms
    của chính fact đó, không phải theo số vế đã trích dẫn — nếu nửa câu
    mô hình thực sự viết ra không phủ hết từ khoá của một fact (mô hình
    ghép câu ở chỗ NÓ chọn, không nhất thiết đúng ranh giới required_fact),
    fact đó vẫn 0 điểm dù trích dẫn đúng. Trên `pub-04-lam-viec-tu-xa` cụ
    thể, trần recall là 0.5 với MỌI harness đúng luật, vì đúng lý do đó —
    đo được, không phải suy đoán. Vẫn nên làm: `abstain: true` sau khi nêu
    cả hai phía được 0.5 recall + trọn 15 điểm honesty, và điểm recall lấy
    theo `max(...)` nên làm cả hai không bao giờ THIỆT — chỉ đừng trông
    đợi nó vượt sàn 0.5 trên brief này.
  * Xoá claim là hợp lệ. SỬA CHỮ trong `claim["text"]` thì KHÔNG: thêm
    một dấu chấm cuối câu cũng đủ làm claim mất cả provenance lẫn hỗ trợ
    (đo được: -40 điểm). Chỉ được xoá, giữ nguyên, hoặc cắt bớt.

GỢI Ý cho trường hợp (c): câu bị ghép là hai đoạn DO CHÍNH MÔ HÌNH viết,
dán với nhau bằng một liên từ (" và "). Cắt đúng chỗ dán thì hai nửa vẫn
là chữ của mô hình — vẫn qua được kiểm tra provenance. Muốn biết cắt đúng
chưa: cả hai nửa phải xuất hiện nguyên văn trong `ctx.observed_text` và
phải thuộc HAI tài liệu khác nhau. Cắt sai thì một nửa sẽ vắt qua hai tài
liệu và không quan sát nào chứa nó.

CÔNG CỤ CÓ SẴN:
    ctx.observed_text  -> toàn bộ quan sát agent đã thấy, nối lại
    ctx.saw(text)      -> text có trong quan sát không
    ctx.corpus.docs    -> danh sách Doc (doc_id, title, body); qua
                          `ctx.corpus`, `Doc.tags` LUÔN RỖNG — CẢ Ở VÒNG
                          LUYỆN TẬP LẪN VÒNG CHẤM ĐIỂM, vì corpus mà code
                          của bạn cầm bị gỡ nhãn bẫy ('outdated',
                          'contradiction', 'injection'…) ngay khi runner
                          dựng lên nó, không phải chỉ lúc chấm điểm. Đọc
                          nhãn là tra bảng chứ không phải kỹ năng lab này
                          chấm. Ở vòng LUYỆN TẬP seed 42 thì file TRÊN ĐĨA
                          `data/corpus/*.json` (khác với `ctx.corpus`)
                          vẫn có nhãn: hard-code được từ đó, và điều đó
                          được nói thẳng ra ở đây thay vì giấu đi.
    ctx.state          -> dict tuỳ bạn dùng để ghi số liệu gỡ lỗi

Cài đặt:  ReActAgent(..., middleware=[InjectionGuard(), Critic(), ...])
Xem `harness/middleware.py` để biết thứ tự các hook.
"""

from __future__ import annotations

import json
import re
import unicodedata

from harness.middleware import Middleware

_WS_RE = re.compile(r"\s+")

#: Danh sách các liên từ / dấu nối thường gặp khi mô hình ghép hai vế mâu thuẫn (Việt & Anh)
CONNECTORS = (
    " và ",
    " and ",
    ", nhưng ",
    " nhưng ",
    ", but ",
    " but ",
    ", tuy nhiên ",
    " tuy nhiên ",
    ", however, ",
    ", trong khi ",
    " trong khi ",
    ", while ",
    " while ",
    "; ",
    " / ",
    ", ",
    " hoặc ",
    " or ",
    " vs ",
    " versus ",
)

MAX_CLAIM_CHARS = 500
MIN_SUPPORT_CHARS = 12
MAX_CLAIMS_PER_DOC = 4
MAX_SCORED_CLAIMS = 10


def _norm(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return _WS_RE.sub(" ", unicodedata.normalize("NFC", text).casefold()).strip()


def _line_supports(doc_body: str, text: str) -> bool:
    if not doc_body or not text:
        return False
    lines = doc_body.splitlines()
    if any(text in line for line in lines):
        return True
    norm_text = _norm(text)
    if len(norm_text) >= MIN_SUPPORT_CHARS:
        return any(norm_text in _norm(line) for line in lines)
    return False


def _find_doc_for_span(corpus, span: str, observed: str, retrieved_ids: set[str]) -> str | None:
    """Tìm doc_id hợp lệ đã quan sát hỗ trợ đoạn text span."""
    if not corpus:
        return None
    for doc in corpus.docs:
        is_retrieved = (
            doc.doc_id in retrieved_ids
            or doc.body in observed
            or any(len(l) >= 25 and l in observed for l in doc.body.splitlines())
        )
        if is_retrieved and _line_supports(doc.body, span):
            return doc.doc_id
    return None


class Critic(Middleware):
    """Xoá những gì bằng chứng không đỡ; giải mâu thuẫn; abstain khi không còn gì."""

    name = "critic"

    def wrap_tool_call(self, ctx, call, name, args):
        result = call(name, args)
        if hasattr(ctx, "state") and isinstance(ctx.state, dict):
            retrieved = ctx.state.setdefault("retrieved_doc_ids", set())
            if name == "fetch_doc":
                doc_id = args.get("doc_id") if isinstance(args, dict) else (args[0] if args else None)
                if isinstance(doc_id, str) and doc_id:
                    retrieved.add(doc_id)
            elif name == "search" and result.ok and isinstance(result.content, str):
                try:
                    items = json.loads(result.content)
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, dict) and "doc_id" in item:
                                retrieved.add(item["doc_id"])
                except Exception:
                    pass
        return result

    def after_agent(self, ctx, report):
        if not isinstance(report, dict):
            return report
        claims = report.get("claims")
        if not isinstance(claims, list) or not claims:
            report["abstain"] = True
            report["claims"] = []
            report["citations"] = []
            report["answer"] = "Không đủ căn cứ để trả lời câu hỏi."
            return report

        observed = ctx.observed_text
        retrieved_ids = set()
        if hasattr(ctx, "state") and isinstance(ctx.state, dict):
            retrieved_ids = ctx.state.get("retrieved_doc_ids", set())

        raw_candidates = []

        for claim in claims:
            if not isinstance(claim, dict):
                continue
            text = claim.get("text")
            if not isinstance(text, str) or not text:
                continue

            # Kiểm tra độ dài cơ bản (scorer loại bỏ < 12 ký tự)
            if len(_norm(text)) < MIN_SUPPORT_CHARS:
                continue

            # Trường hợp 1: text nằm trong observed (trực tiếp hoặc sau chuẩn hoá Unicode/khoảng trắng)
            if text in observed or _norm(text) in _norm(observed):
                raw_candidates.append(claim)
                continue

            # Trường hợp 2: câu ghép mâu thuẫn (contradiction fusion)
            split_success = False

            # Thử các liên từ / dấu nối thông dụng
            for conn in CONNECTORS:
                if conn in text:
                    parts = text.split(conn)
                    for i in range(1, len(parts)):
                        left = conn.join(parts[:i]).strip()
                        right = conn.join(parts[i:]).strip()
                        if (
                            len(left) >= MIN_SUPPORT_CHARS
                            and len(right) >= MIN_SUPPORT_CHARS
                            and left in observed
                            and right in observed
                        ):
                            doc_left = _find_doc_for_span(ctx.corpus, left, observed, retrieved_ids)
                            doc_right = _find_doc_for_span(ctx.corpus, right, observed, retrieved_ids)
                            if doc_left and doc_right and doc_left != doc_right:
                                raw_candidates.append({"text": left, "doc_id": doc_left})
                                raw_candidates.append({"text": right, "doc_id": doc_right})
                                report["abstain"] = True
                                split_success = True
                                break
                    if split_success:
                        break

            # Nếu liên từ cố định chưa tách được, thử cắt tại các ranh giới từ
            if not split_success and len(text) >= 24:
                # Quét các điểm cắt hợp lệ
                for i in range(MIN_SUPPORT_CHARS, len(text) - MIN_SUPPORT_CHARS):
                    if text[i] in " ,;.-/":
                        left = text[:i].strip()
                        right = text[i + 1:].strip()
                        if (
                            len(left) >= MIN_SUPPORT_CHARS
                            and len(right) >= MIN_SUPPORT_CHARS
                            and left in observed
                            and right in observed
                        ):
                            doc_left = _find_doc_for_span(ctx.corpus, left, observed, retrieved_ids)
                            doc_right = _find_doc_for_span(ctx.corpus, right, observed, retrieved_ids)
                            if doc_left and doc_right and doc_left != doc_right:
                                raw_candidates.append({"text": left, "doc_id": doc_left})
                                raw_candidates.append({"text": right, "doc_id": doc_right})
                                report["abstain"] = True
                                split_success = True
                                break

        # Lọc, chuẩn hoá giới hạn và chống spam claim
        seen_pairs = set()
        per_doc_count: dict[str, int] = {}
        filtered_claims = []

        for c in raw_candidates:
            t = c.get("text")
            d = c.get("doc_id")
            if not isinstance(t, str) or not isinstance(d, str):
                continue
            pair = (t, d)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            if per_doc_count.get(d, 0) >= MAX_CLAIMS_PER_DOC:
                continue
            if len(filtered_claims) >= MAX_SCORED_CLAIMS:
                break

            per_doc_count[d] = per_doc_count.get(d, 0) + 1
            filtered_claims.append(c)

        if not filtered_claims:
            report["abstain"] = True
            report["claims"] = []
            report["citations"] = []
            report["answer"] = "Không đủ căn cứ để trả lời câu hỏi."
        else:
            report["claims"] = filtered_claims
            report["citations"] = sorted(set(
                c["doc_id"] for c in filtered_claims if isinstance(c, dict) and c.get("doc_id")
            ))

        return report

