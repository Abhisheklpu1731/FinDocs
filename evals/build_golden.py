import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

load_dotenv()

PAGES_PATH = Path("data/processed/pages.jsonl")
OUT_PATH = Path("evals/golden.jsonl")
PER_DOC = 20          # pages to try per company
MIN_CHARS = 800       # skip near-empty pages
PAUSE = 2.5           # seconds between calls, to stay under free-tier limits
COMPANY = {"hdfc_bank_2025": "HDFC", "infosys_2025": "Infosys", "lenskart_2025": "Lenskart"}

DRAFT_PROMPT = """You write test questions for a question-answering system over company annual reports.
Read the page and write ONE question a business user might ask, whose answer is clearly stated on this page.
Rules:
- Include the company name in the question.
- The answer must be a short, specific fact or number, copied EXACTLY from the page.
- If the page has no clear, specific fact, reply with exactly: SKIP
Reply ONLY with JSON like: {"question": "...", "answer": "..."}"""

PARAPHRASE_PROMPT = """Rewrite this question the way a busy person would type it into a search box.
Use casual, everyday words. Keep the company name and exactly what is being asked. Do not add facts.
Reply with ONLY the rewritten question."""

UNANSWERABLE_PROMPT = """We have only the FY2024-25 annual reports of HDFC Bank, Infosys and Lenskart.
Write 10 realistic questions a user might ask that CANNOT be answered from those reports.
Mix: other companies, other years (e.g. FY2019), future predictions, and stock price questions.
Reply ONLY with a JSON list of strings."""

DATE_ONLY = re.compile(r"^\W*(\d{1,2}(st|nd|rd|th)?\s+\w+,?\s+\d{4}|\w+\s+\d{1,2},?\s+\d{4})\W*$")


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def ask(client, model, system: str, user: str) -> str:
    """One LLM call, with a pause and simple retry if we hit the rate limit."""
    for _ in range(3):
        try:
            response = client.chat.completions.create(
                model=model, temperature=0,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
            )
            time.sleep(PAUSE)
            return response.choices[0].message.content.strip()
        except RateLimitError:
            print("   rate limited, waiting 30s...")
            time.sleep(30)
    return ""


def parse_json(text: str):
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def main() -> None:
    client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1")
    model = os.environ["LLM_MODEL"]
    rng = random.Random(42)

    with PAGES_PATH.open(encoding="utf-8") as f:
        pages = [json.loads(line) for line in f]
    pages = [p for p in pages if not p["scanned"] and p["n_chars"] >= MIN_CHARS]

    golden, dropped = [], Counter()

    for doc_id, company in COMPANY.items():
        doc_pages = [p for p in pages if p["doc_id"] == doc_id]
        for page in rng.sample(doc_pages, PER_DOC):
            qa = parse_json(ask(client, model, DRAFT_PROMPT,
                                f"Document: {doc_id}\n\n{page['text'][:4000]}"))
            if not isinstance(qa, dict) or "answer" not in qa:
                dropped["skipped or bad JSON"] += 1
                continue
            if norm(qa["answer"]) not in norm(page["text"]):
                dropped["answer not on page"] += 1
                continue
            if DATE_ONLY.match(qa["answer"]):
                dropped["trivial date"] += 1
                continue

            question = ask(client, model, PARAPHRASE_PROMPT, qa["question"])
            if company.lower() not in question.lower():
                dropped["company name lost (garbled?)"] += 1
                continue

            golden.append({
                "id": f"g{len(golden) + 1:03d}",
                "question": question,
                "draft_question": qa["question"],
                "answer": qa["answer"],
                "doc_id": doc_id,
                "pages": [page["page"]],
                "type": "number" if any(ch.isdigit() for ch in qa["answer"]) else "fact",
            })
            print(f"kept  {doc_id} p{page['page']}: {question}")

    questions = parse_json(ask(client, model, UNANSWERABLE_PROMPT, "Write them now.")) or []
    for q in questions:
        golden.append({"id": f"g{len(golden) + 1:03d}", "question": q, "answer": "NOT_FOUND",
                       "doc_id": None, "pages": [], "type": "unanswerable"})

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for item in golden:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nKept {len(golden)} questions ({len(questions)} unanswerable) -> {OUT_PATH}")
    print("Dropped:", dict(dropped))


if __name__ == "__main__":
    main()