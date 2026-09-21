"""End-to-end RAG pipeline: retrieve then generate."""
from dataclasses import dataclass

from .generator import Generator
from .retriever import Retriever


@dataclass
class RAGResult:
    answer: str
    contexts: list[dict]


class RAGPipeline:
    def __init__(self, top_k: int = 5, generator_api_key_env: str = "GEMINI_API_KEY") -> None:
        self.retriever = Retriever()
        self.generator = Generator(api_key_env=generator_api_key_env)
        self.top_k = top_k

    def run(self, question: str, top_k: int | None = None) -> RAGResult:
        contexts = self.retriever.retrieve(question, top_k=top_k or self.top_k)
        answer = self.generator.generate(question, contexts)
        return RAGResult(answer=answer, contexts=contexts)


if __name__ == "__main__":
    import sys

    pipeline = RAGPipeline()
    question = " ".join(sys.argv[1:]) or "How do I define a path parameter with a type?"
    result = pipeline.run(question)
    print("ANSWER:\n", result.answer)
    print("\nSOURCES:")
    for c in result.contexts:
        print(f"  [{c['score']:.3f}] {c['source']} — {c['heading_path']}")
