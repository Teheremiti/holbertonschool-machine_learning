#!/usr/bin/env python3
"""
Bag of Words embedding implementation.

This module implements a bag-of-words embedding function that creates
count-based embeddings from text sentences.
"""
import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """
    Create a bag of words embedding matrix.

    Args:
        sentences (list): List of sentences to analyze.
        vocab (list, optional): List of vocabulary words to use for analysis.
                                If None, all words within sentences are used.

    Returns:
        tuple: (embeddings, features)
            - embeddings: numpy.ndarray of shape (s, f) containing the embeddings
              where s is the number of sentences and f is the number of features
            - features: list of the features used for embeddings
    """
    # Tokenize sentences: lowercase, remove punctuation, split by whitespace
    tokenized_sentences = []
    for sentence in sentences:
        # Convert to lowercase and remove punctuation
        cleaned = re.sub(r'[^\w\s]', '', sentence.lower())
        # Split by whitespace and filter empty strings
        tokens = [token for token in cleaned.split() if token]
        tokenized_sentences.append(tokens)

    # Build vocabulary
    if vocab is None:
        # Collect all unique words from all sentences
        all_words = set()
        for tokens in tokenized_sentences:
            all_words.update(tokens)
        features = sorted(list(all_words))
    else:
        # Use provided vocabulary
        features = sorted(list(set(vocab)))

    # Create embedding matrix
    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f), dtype=int)

    # Create word to index mapping for faster lookup
    word_to_idx = {word: idx for idx, word in enumerate(features)}

    # Count word occurrences for each sentence
    for i, tokens in enumerate(tokenized_sentences):
        for token in tokens:
            if token in word_to_idx:
                embeddings[i, word_to_idx[token]] += 1

    return embeddings, features
