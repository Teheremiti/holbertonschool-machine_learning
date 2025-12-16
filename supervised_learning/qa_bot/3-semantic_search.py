#!/usr/bin/python3
"""Semantic search utility built on sentence-transformers."""
import os
from sentence_transformers import SentenceTransformer, SimilarityFunction
import numpy as np

_SEMANTIC_MODEL = None


def _get_semantic_model():
    """Return a cached sentence-transformer model configured for cosine."""
    global _SEMANTIC_MODEL
    if _SEMANTIC_MODEL is None:
        model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
        model.similarity_fn_name = SimilarityFunction.COSINE
        _SEMANTIC_MODEL = model
    return _SEMANTIC_MODEL


def semantic_search(corpus_path, sentence):
    """Perform semantic search on a corpus of markdown documents.

    This function loads all ``.md`` files from ``corpus_path`` and uses a
    sentence-transformer model to select the document that is most
    similar to ``sentence`` under cosine similarity.

    Args:
        corpus_path (str): Path to the corpus of reference documents.
        sentence (str): Query sentence used for semantic search.

    Returns:
        str: The reference text of the document most similar to
        ``sentence``.
    """
    corpus = []
    for filename in os.listdir(corpus_path):
        if filename.endswith(".md"):
            file_path = os.path.join(corpus_path, filename)
            with open(file_path, "r", encoding="utf-8") as md_file:
                corpus.append(md_file.read() + "\n")

    model = _get_semantic_model()

    # Encode the query sentence and all corpus documents.
    embedding_sentence = model.encode(sentence)
    embeddings_corpus = model.encode(corpus)

    similarities = [
        model.similarity(embedding_sentence, embedding_doc)
        for embedding_doc in embeddings_corpus
    ]

    best_similarity_idx = np.argmax(similarities)
    best_doc = corpus[best_similarity_idx]

    return best_doc
