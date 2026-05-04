from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json_array(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return data


def load_articles(config: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for law in load_json_array(config.corpus_path):
        law_id = law.get(config.corpus_law_id_field)
        articles = law.get(config.corpus_articles_field, [])
        if not isinstance(articles, list):
            raise ValueError(f"Expected article list in field {config.corpus_articles_field!r}")
        for article in articles:
            aid = article.get(config.article_id_field)
            text = article.get(config.article_text_field, "")
            if aid is None:
                continue
            rows.append({"aid": str(aid), "law_id": str(law_id), "text": str(text or "")})
    return rows


def load_questions(config: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in load_json_array(config.questions_path):
        qid = item.get(config.question_id_field)
        question = item.get(config.question_text_field, "")
        relevant = item.get(config.relevant_ids_field, [])
        if qid is None:
            continue
        if not isinstance(relevant, list):
            relevant = [relevant]
        rows.append(
            {
                "qid": str(qid),
                "question": str(question or ""),
                "relevant_laws": [str(value) for value in relevant],
            }
        )
    return rows
