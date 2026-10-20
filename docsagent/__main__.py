"""Command line: python -m docsagent ingest sample_docs  |  python -m docsagent search "question"."""
from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="docsagent",
                                     description="Answer questions from your documents, with sources.")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest", help="Read a folder of documents into the vector database")
    ingest.add_argument("folder")
    search = sub.add_parser("search", help="Show the document pieces closest to a question")
    search.add_argument("question")
    search.add_argument("-k", type=int, default=4, help="How many pieces to show (default 4)")
    ask = sub.add_parser("ask", help="Answer a question with a source for every sentence (needs ANTHROPIC_API_KEY)")
    ask.add_argument("question")
    evaluate = sub.add_parser("eval", help="Score the agent on evals/questions.jsonl")
    evaluate.add_argument("--retrieval-only", action="store_true", help="Score search only (free, no API key)")
    evaluate.add_argument("--save", metavar="FILE", help="Also write the report to a Markdown file")
    args = parser.parse_args(argv)

    if args.command == "ask":
        return _ask(args.question)
    if args.command == "eval":
        return _eval(args.retrieval_only, args.save)

    from .chunking import load_folder
    from .vectorstore import FastEmbedder, VectorStore

    store = VectorStore(FastEmbedder())
    if args.command == "ingest":
        chunks = load_folder(args.folder)
        store.add(chunks)
        files = len({c.source for c in chunks})
        print(f"Stored {len(chunks)} chunks from {files} files. The database now holds {store.count()} chunks.")
    elif args.command == "search":
        for chunk, score in store.search(args.question, args.k):
            preview = " ".join(chunk.text.split())[:150]
            print(f"{score:.3f}  [{chunk.label()}]  {preview}")


def _ask(question: str) -> None:
    from .agent import ClaudeLLM, ask, build_agent
    from .vectorstore import FastEmbedder, VectorStore

    state = ask(build_agent(VectorStore(FastEmbedder()), ClaudeLLM()), question)
    checked, hits = state["checked"], state["hits"]
    print(f"Answer: {checked.text}\n")
    if checked.cited:
        print("Sources:")
        for n in checked.cited:
            chunk, score = hits[n - 1]
            print(f"  [{n}] {chunk.label()}  (match {score:.2f})")
    if checked.dropped:
        print(f"\nRemoved {len(checked.dropped)} sentence(s) without support:")
        for sentence, reason in checked.dropped:
            print(f"  - {sentence}  ({reason})")
    print("\nSteps: " + " → ".join(state["steps"]))


def _eval(retrieval_only: bool, save: str | None) -> None:
    from .evaluate import load_cases, report, run
    from .vectorstore import FastEmbedder, VectorStore

    store = VectorStore(FastEmbedder())
    agent = None
    if not retrieval_only:
        from .agent import ClaudeLLM, build_agent
        agent = build_agent(store, ClaudeLLM())
    text = report(run(load_cases(), store, agent))
    print(text)
    if save:
        with open(save, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"\nSaved to {save}")


if __name__ == "__main__":
    main()
