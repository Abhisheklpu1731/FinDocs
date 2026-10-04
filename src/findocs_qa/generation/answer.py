import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

from findocs_qa.retrieval.rerank import RerankRetriever

load_dotenv()

SYSTEM_PROMPT = """You answer questions about company annual reports using ONLY the sources provided.

Rules:
1. Use only information found in the sources. Never use outside knowledge.
2. After every fact, cite its source as [doc_id, page N].
3. If the sources do not contain the answer, reply exactly: Not found in the documents.
4. Copy numbers and units exactly as written (for example "2,14,552" or "INR Million").
5. In some sources a backtick (`) stands for the rupee symbol (₹).
6. Keep the answer short: one to three sentences."""


class Answerer:
    """Retrieves the best chunks, then asks the LLM to answer from them with citations."""

    def __init__(self, strategy: str = "recursive", retriever=None):
        self.retriever = retriever or RerankRetriever(strategy)
        self.client = OpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
        )
        self.model = os.environ["LLM_MODEL"]

    def answer(
        self, question: str, k: int = 5, doc_ids: set[str] | None = None, on_step=None
    ) -> dict:
        """on_step(name, detail) reports progress: hybrid -> rerank -> generate."""
        chunks = self.retriever.search(question, k=k, doc_ids=doc_ids, on_step=on_step)
        if on_step:
            on_step("generate", f"asking {self.model}")

        sources = "\n\n".join(
            f'<source doc_id="{c["doc_id"]}" page="{c["page"]}">\n{c["text"]}\n</source>'
            for c in chunks
        )

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{sources}\n\nQuestion: {question}"},
            ],
        )

        text = response.choices[0].message.content.replace("【", "[").replace("】", "]")

        return {
            "question": question,
            "answer": text,
            "sources": [(c["doc_id"], c["page"]) for c in chunks],
            "chunks": chunks,
            "tokens": response.usage.total_tokens,
        }


if __name__ == "__main__":
    answerer = Answerer()
    question = " ".join(sys.argv[1:]) or "Who is the CEO of Infosys?"
    result = answerer.answer(question)
    print(f"Q: {result['question']}\n")
    print(f"A: {result['answer']}\n")
    print(f"Retrieved: {result['sources']}")
    print(f"Tokens used: {result['tokens']}")