import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PAGES_PATH = Path("data/processed/pages.jsonl")
OUT_PATH = Path("evals/candidates.jsonl")
PER_DOC = 10        # questions to draft per company
MIN_CHARS = 800     # skip near-empty pages
random.seed(42)     # same random pages every run

PROMPT = """You write test questions for a question-answering system over company annual reports.
Read the page and write ONE question a business user might ask, whose answer is clearly stated on this page.

Rules:
- Use plain, everyday wording. Do not copy the page's phrasing.
- Include the company name in the question.
- The answer must be a specific fact or number, copied exactly from the page.
- If the page has no clear, specific fact, reply with exactly: SKIP

Reply ONLY with JSON like: {"question": "...", "answer": "..."}"""


def load_pages() -> list[dict]:
    with PAGES_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    client = OpenAI(api_key=os.environ["GROQ_API_KEY"],
                    base_url="https://api.groq.com/openai/v1")
    model = os.environ["LLM_MODEL"]

    pages = [p for p in load_pages() if not p["scanned"] and p["n_chars"] >= MIN_CHARS]
    doc_ids = sorted({p["doc_id"] for p in pages})

    count = 0
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for doc_id in doc_ids:
            doc_pages = [p for p in pages if p["doc_id"] == doc_id]
            for page in random.sample(doc_pages, PER_DOC):
                response = client.chat.completions.create(
                    model=model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": PROMPT},
                        {"role": "user", "content": f"Document: {doc_id}\n\n{page['text'][:6000]}"},
                    ],
                )
                text = response.choices[0].message.content.strip()
                text = text.removeprefix("```json").removesuffix("```").strip()

                if text == "SKIP":
                    print(f"skip   {doc_id} page {page['page']}")
                    continue
                try:
                    qa = json.loads(text)
                except json.JSONDecodeError:
                    print(f"badjson {doc_id} page {page['page']}")
                    continue

                count += 1
                record = {
                    "id": f"c{count:03d}",
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "doc_id": doc_id,
                    "pages": [page["page"]],
                    "type": "number" if any(ch.isdigit() for ch in qa["answer"]) else "fact",
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"ok     {doc_id} page {page['page']}: {qa['question']}")

    print(f"\nDrafted {count} candidates -> {OUT_PATH}")


if __name__ == "__main__":
    main()