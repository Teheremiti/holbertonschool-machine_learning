#!/usr/bin/python3
"""Multi-reference question answering using semantic search and BERT."""
from sentence_transformers import SentenceTransformer, SimilarityFunction
import os
import numpy as np

# Lazily initialized global models reused across calls.
semantic_model = None

# Reuse the single-reference QA implementation from ``0-qa.py``.
question_answer = __import__("0-qa").question_answer


def load_corpus(corpus_path):
    """Load all markdown documents from a corpus directory.

    Args:
        corpus_path (str): Path to the corpus of reference documents.

    Returns:
        list[str]: List of document contents, one per ``.md`` file,
        each ending with a newline.
    """
    corpus = []
    for filename in os.listdir(corpus_path):
        if filename.endswith(".md"):
            file_path = os.path.join(corpus_path, filename)
            with open(file_path, "r", encoding="utf-8") as md_file:
                corpus.append(md_file.read() + "\n")
    return corpus


def semantic_search(corpus_path, sentence):
    """Perform semantic search on a corpus of documents.

    The function reads all markdown files in ``corpus_path`` and uses a
    sentence-transformer model to select the document that is most
    similar to ``sentence``. If the maximum similarity score does not
    pass a heuristic threshold, None is returned.

    Args:
        corpus_path (str): Path to the corpus of reference documents.
        sentence (str): Sentence from which to perform semantic search.

    Returns:
        str or None: Reference text of the most similar document, or
        None when the similarity threshold is not met.
    """
    corpus = load_corpus(corpus_path)

    # Lazily load and configure the semantic model.
    global semantic_model
    if semantic_model is None:
        semantic_model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
        semantic_model.similarity_fn_name = SimilarityFunction.COSINE

    # Encode input question.
    embeddings_sen = semantic_model.encode(sentence)

    similarities = [
        semantic_model.similarity(embeddings_sen, semantic_model.encode(doc))
        for doc in corpus
    ]

    best_similarity_idx = np.argmax(similarities)
    if np.argmax(similarities) > 0.5:
        best_doc = corpus[best_similarity_idx]
    else:
        best_doc = None

    return best_doc


def question_answer(corpus_path):
    """Answer questions from multiple reference texts in a loop.

    The function repeatedly prompts the user for a question using the
    prefix ``Q: ``. It first performs semantic search over the corpus
    located at ``corpus_path`` to pick the most relevant document, then
    applies :func:`q_a` to extract an answer span.

    Typing any of ``\"exit\"``, ``\"quit\"``, ``\"goodbye\"``, or
    ``\"bye\"`` (case-insensitive) prints ``A: Goodbye`` and stops the
    loop.

    Args:
        corpus_path (str): Path to the corpus of reference documents.
    """
    while True:
        question = input("Q: ")
        if question.lower() in ["exit", "quit", "goodbye", "bye"]:
            print("A: Goodbye")
            break

        # First semantic search to pick a relevant reference.
        reference = semantic_search(corpus_path, question)

        # Second step: question answering on the selected reference.
        answer = question_answer(question, reference)

        if not answer:
            print("A: Sorry, I do not understand your question.")
        else:
            print(f"A: {answer}")
