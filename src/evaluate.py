"""
Evaluation harness for the RAG pipeline.

Two things get measured, because "it gives good-sounding answers" is not
evidence of anything:

1. RETRIEVAL QUALITY (reference-based, no LLM needed):
   - Recall@k: did the correct source document appear in the top-k retrieved chunks?
   - MRR (Mean Reciprocal Rank): how high up was the first correct chunk?

2. ANSWER QUALITY (LLM-as-judge, requires an LLM provider):
   - Faithfulness: is the generated answer actually supported by the retrieved
     context, or did the model hallucinate beyond it?
   - Relevance: does the answer actually address the question asked?

Run with:
    python -m src.evaluate
"""
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

from src.config import settings, EVAL_DIR, PROJECT_ROOT
from src.rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

EVAL_SET_PATH = PROJECT_ROOT / "eval" / "eval_set.json"


@dataclass
class RetrievalResult:
    question: str
    expected_source: str
    retrieved_sources: List[str]
    hit_at_k: bool
    reciprocal_rank: float


@dataclass
class AnswerJudgment:
    question: str
    answer: str
    faithfulness_score: int  # 1-5
    relevance_score: int     # 1-5
    reasoning: str


def load_eval_set() -> List[dict]:
    with open(EVAL_SET_PATH) as f:
        return json.load(f)


def evaluate_retrieval(pipeline: RAGPipeline, eval_set: List[dict]) -> List[RetrievalResult]:
    results = []
    for item in eval_set:
        chunks = pipeline.retrieve(item["question"], top_k=settings.top_k)
        retrieved_sources = [c.source for c in chunks]

        hit = item["expected_source"] in retrieved_sources
        rr = 0.0
        if item["expected_source"] in retrieved_sources:
            rank = retrieved_sources.index(item["expected_source"]) + 1
            rr = 1.0 / rank

        results.append(RetrievalResult(
            question=item["question"],
            expected_source=item["expected_source"],
            retrieved_sources=retrieved_sources,
            hit_at_k=hit,
            reciprocal_rank=rr,
        ))
    return results


def summarize_retrieval(results: List[RetrievalResult]) -> dict:
    n = len(results)
    recall_at_k = sum(r.hit_at_k for r in results) / n if n else 0.0
    mrr = sum(r.reciprocal_rank for r in results) / n if n else 0.0
    return {"n_questions": n, "recall_at_k": round(recall_at_k, 3), "mrr": round(mrr, 3)}


JUDGE_PROMPT_TEMPLATE = """You are evaluating a RAG system's answer for quality. Be strict.

Question: {question}

Retrieved context the system had access to:
{context}

System's answer:
{answer}

Score the answer on two dimensions, each from 1 (worst) to 5 (best):
- faithfulness: Is every claim in the answer actually supported by the retrieved context?
  A score of 1 means the answer contains claims not found in the context (hallucination).
  A score of 5 means every claim traces directly back to the context.
- relevance: Does the answer actually address what was asked?

Respond ONLY with valid JSON in this exact format, no other text:
{{"faithfulness_score": <int 1-5>, "relevance_score": <int 1-5>, "reasoning": "<one sentence>"}}
"""


def judge_answer(pipeline: RAGPipeline, question: str) -> AnswerJudgment:
    """Use the same LLM provider as a judge. In a larger project you'd use a
    stronger/independent model as judge to avoid self-preference bias."""
    response = pipeline.answer(question)
    context = "\n\n".join(c.text for c in response.retrieved_chunks)
    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question, context=context, answer=response.answer
    )

    raw = pipeline._generate_anthropic(judge_prompt) if settings.llm_provider == "anthropic" \
        else pipeline._generate_openai(judge_prompt)

    try:
        parsed = json.loads(raw.strip().strip("```json").strip("```"))
    except json.JSONDecodeError:
        logger.warning("Judge did not return valid JSON, defaulting to 0 scores: %s", raw[:200])
        parsed = {"faithfulness_score": 0, "relevance_score": 0, "reasoning": "parse_error"}

    return AnswerJudgment(
        question=question,
        answer=response.answer,
        faithfulness_score=parsed.get("faithfulness_score", 0),
        relevance_score=parsed.get("relevance_score", 0),
        reasoning=parsed.get("reasoning", ""),
    )


def run_full_evaluation():
    pipeline = RAGPipeline()
    eval_set = load_eval_set()

    logger.info("Running retrieval evaluation on %d questions...", len(eval_set))
    retrieval_results = evaluate_retrieval(pipeline, eval_set)
    retrieval_summary = summarize_retrieval(retrieval_results)
    logger.info("Retrieval summary: %s", retrieval_summary)

    report = {
        "retrieval_summary": retrieval_summary,
        "retrieval_details": [asdict(r) for r in retrieval_results],
    }

    if settings.llm_provider in ("anthropic", "openai") and (
        settings.anthropic_api_key or settings.openai_api_key
    ):
        logger.info("Running LLM-as-judge answer evaluation...")
        judgments = [judge_answer(pipeline, item["question"]) for item in eval_set]
        avg_faithfulness = sum(j.faithfulness_score for j in judgments) / len(judgments)
        avg_relevance = sum(j.relevance_score for j in judgments) / len(judgments)
        report["answer_summary"] = {
            "avg_faithfulness": round(avg_faithfulness, 2),
            "avg_relevance": round(avg_relevance, 2),
        }
        report["answer_details"] = [asdict(j) for j in judgments]
    else:
        logger.info("Skipping answer evaluation: no LLM API key configured.")

    out_path = EVAL_DIR / "eval_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Full evaluation report written to %s", out_path)
    return report


if __name__ == "__main__":
    run_full_evaluation()
