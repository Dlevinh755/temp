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
    parser.add_argument("--num_extra_laws", type=int, default=100)
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

    positive_laws = []
    extra_candidates = []
    for law in corpus:
        if law_aids(law) & positive_aids:
            positive_laws.append(law)
        else:
            extra_candidates.append(law)

    num_extra_laws = min(args.num_extra_laws, len(extra_candidates))
    sampled_corpus = positive_laws + rng.sample(extra_candidates, num_extra_laws)
    sampled_corpus.sort(key=lambda item: str(item.get("law_id", item.get("id", ""))))
    sampled_questions.sort(key=lambda item: str(item.get("qid", "")))

    write_json(args.output_dir / args.corpus_file, sampled_corpus)
    write_json(args.output_dir / args.questions_file, sampled_questions)
    write_json(
        args.output_dir / "sample_manifest.json",
        {
            "source_dir": str(args.input_dir),
            "num_questions": len(sampled_questions),
            "num_laws": len(sampled_corpus),
            "num_positive_laws": len(positive_laws),
            "num_extra_laws": num_extra_laws,
            "seed": args.seed,
            "corpus_file": args.corpus_file,
            "questions_file": args.questions_file,
        },
    )
    print(f"[save] {args.output_dir / args.corpus_file}")
    print(f"[save] {args.output_dir / args.questions_file}")
    print(f"[done] questions={len(sampled_questions)} laws={len(sampled_corpus)}")


if __name__ == "__main__":
    main()
