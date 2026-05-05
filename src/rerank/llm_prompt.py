from __future__ import annotations

from typing import Any


LLM_RERANK_SYSTEM_PROMPT = (
    "Bạn là một chuyên gia đánh giá mức độ liên quan của văn bản pháp luật.\n\n"
    "Nhiệm vụ của bạn là xác định xem đoạn văn bản được cung cấp có liên quan trực tiếp đến câu hỏi hay không.\n\n"
    "Quy ước nhãn:\n"
    "1 = Có liên quan. Đoạn văn bản chứa thông tin pháp lý có thể dùng để trả lời hoặc hỗ trợ trả lời câu hỏi.\n"
    "0 = Không liên quan. Đoạn văn bản không liên quan, chỉ liên quan rất gián tiếp, hoặc không cung cấp thông tin đủ để trả lời câu hỏi.\n\n"
    "Yêu cầu bắt buộc:\n"
    "- Chỉ trả về duy nhất một chữ số: 0 hoặc 1.\n"
    "- Không giải thích.\n"
    "- Không thêm bất kỳ ký tự, dấu câu, khoảng trắng, markdown hoặc văn bản nào khác."
)


def build_llm_rerank_messages(question: str, chunk_text: str, label: int | None = None) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": LLM_RERANK_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Câu hỏi:\n"
                f"{question}\n\n"
                "Đoạn văn bản pháp luật:\n"
                f"{chunk_text}\n\n"
                "Đoạn văn bản trên có chứa thông tin pháp lý có thể dùng để trả lời hoặc hỗ trợ trả lời trực tiếp câu hỏi không?\n"
                "Chỉ trả lời 0 hoặc 1."
            ),
        },
    ]
    if label is not None:
        messages.append({"role": "assistant", "content": str(int(label))})
    return messages


def build_llm_train_rows(ready_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in ready_rows:
        for passage in row.get("passages", []):
            rows.append(
                {
                    "qid": str(row["qid"]),
                    "query": str(row["query"]),
                    "chunk_id": str(passage["chunk_id"]),
                    "aid": str(passage["aid"]),
                    "text": str(passage["text"]),
                    "label": int(passage["label"]),
                }
            )
    return rows
