from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a path-compatible sample of legalraw.")
    parser.add_argument("--input_dir", type=Path, default=Path("raw_data/legalraw/full"))
    parser.add_argument("--output_dir", type=Path, default=Path("raw_data/legalraw/sample"))
    parser.add_argument("--num_questions", type=int, default=100)
    parser.add_argument("--num_extra_laws", type=int, default=0, help="Legacy mode: sample this many whole non-positive laws only when --num_extra_articles is 0.")
    parser.add_argument("--num_extra_articles", type=int, default=200, help="Sample this many non-positive articles/aids without pulling whole laws.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corpus_file", default="legal_corpus.json")
    parser.add_argument("--questions_file", default="train.json")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def law_aids(law: dict[str, Any]) -> set[str]:
    return {str(article.get("aid")) for article in law.get("content", []) if article.get("aid") is not None}


def clone_law_with_articles(law: dict[str, Any], articles: list[dict[str, Any]]) -> dict[str, Any]:
    cloned = dict(law)
    cloned["content"] = articles
    return cloned


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    corpus_path = args.input_dir / args.corpus_file
    questions_path = args.input_dir / args.questions_file

    corpus = read_json(corpus_path)
    questions = read_json(questions_path)
    if not isinstance(corpus, list) or not isinstance(questions, list):
        raise ValueError("Both corpus and questions files must contain JSON arrays.")

    num_questions = min(args.num_questions, len(questions))
    sampled_questions = rng.sample(questions, num_questions)
    positive_aids = {
        str(aid)
        for question in sampled_questions
        for aid in question.get("relevant_laws", [])
    }

    positive_articles_by_law: dict[str, list[dict[str, Any]]] = {}
    positive_laws_by_key: dict[str, dict[str, Any]] = {}
    extra_law_candidates = []
    extra_article_candidates: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for law in corpus:
        law_key = str(law.get("law_id", law.get("id", "")))
        articles = law.get("content", [])
        matching_articles = [
            article
            for article in articles
            if str(article.get("aid")) in positive_aids
        ]
        if matching_articles:
            positive_laws_by_key[law_key] = law
            positive_articles_by_law.setdefault(law_key, []).extend(matching_articles)
        else:
            extra_law_candidates.append(law)
        for article in articles:
            if str(article.get("aid")) not in positive_aids:
                extra_article_candidates.append((law_key, law, article))

    if args.num_extra_articles > 0:
        sampled_articles = rng.sample(extra_article_candidates, min(args.num_extra_articles, len(extra_article_candidates)))
        extra_articles_by_law: dict[str, list[dict[str, Any]]] = {}
        extra_laws_by_key: dict[str, dict[str, Any]] = {}
        for law_key, law, article in sampled_articles:
            extra_laws_by_key[law_key] = law
            extra_articles_by_law.setdefault(law_key, []).append(article)
        sampled_corpus = []
        for law_key, law in {**positive_laws_by_key, **extra_laws_by_key}.items():
            articles = positive_articles_by_law.get(law_key, []) + extra_articles_by_law.get(law_key, [])
            sampled_corpus.append(clone_law_with_articles(law, articles))
        num_extra_laws = len(extra_laws_by_key)
        num_extra_articles = len(sampled_articles)
    else:
        positive_laws = [
            clone_law_with_articles(positive_laws_by_key[law_key], articles)
            for law_key, articles in positive_articles_by_law.items()
        ]
        num_extra_laws = min(args.num_extra_laws, len(extra_law_candidates))
        sampled_corpus = positive_laws + rng.sample(extra_law_candidates, num_extra_laws)
        num_extra_articles = sum(len(law.get("content", [])) for law in sampled_corpus) - sum(len(law.get("content", [])) for law in positive_laws)
    sampled_corpus.sort(key=lambda item: str(item.get("law_id", item.get("id", ""))))
    sampled_questions.sort(key=lambda item: str(item.get("qid", "")))
    num_articles = sum(len(law.get("content", [])) for law in sampled_corpus)

    write_json(args.output_dir / args.corpus_file, sampled_corpus)
    write_json(args.output_dir / args.questions_file, sampled_questions)
    write_json(
        args.output_dir / "sample_manifest.json",
        {
            "source_dir": str(args.input_dir),
            "num_questions": len(sampled_questions),
            "num_laws": len(sampled_corpus),
            "num_articles": num_articles,
            "num_positive_laws": len(positive_laws_by_key),
            "num_positive_articles": sum(len(articles) for articles in positive_articles_by_law.values()),
            "num_extra_laws": num_extra_laws,
            "num_extra_articles": num_extra_articles,
            "sample_unit": "article" if args.num_extra_articles > 0 else "law",
            "note": "Positive corpus rows include only sampled qid->relevant aid articles; article mode adds extra individual aids without pulling whole laws.",
            "seed": args.seed,
            "corpus_file": args.corpus_file,
            "questions_file": args.questions_file,
        },
    )
    print(f"[save] {args.output_dir / args.corpus_file}")
    print(f"[save] {args.output_dir / args.questions_file}")
    print(f"[done] questions={len(sampled_questions)} laws={len(sampled_corpus)} articles={num_articles}")


if __name__ == "__main__":
    main()
