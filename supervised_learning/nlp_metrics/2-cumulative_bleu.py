#!/usr/bin/env python3
"""Cumulative BLEU score (up to order n) without external dependencies.

Computes the geometric mean of clipped precisions for 1..n-grams with a
brevity penalty. Inputs are tokenized lists of strings.
"""
from collections import Counter
import numpy as np


def generate_ngram(sentence, order):
    """Generate contiguous n-grams from a tokenized sentence.

    Args:
        sentence (list[str]): Tokenized sentence.
        order (int): N-gram order (n >= 1).

    Returns:
        list[str]: List of n-grams joined by single spaces.
    """
    ngrams = []
    for i in range(len(sentence) - order + 1):
        ngram = sentence[i: i + order]
        ngrams.append(' '.join(ngram))

    return ngrams


def modified_precision(references, sentence, order):
    """Calculate clipped n-gram precision for a given order.

    Args:
        references (list[list[str]]): Reference translations; tokenized.
        sentence (list[str]): Candidate translation; tokenized.
        order (int): N-gram order.

    Returns:
        float: Clipped precision in [0, 1]. Returns 0 if no n-grams exist.
    """
    sentence_ngrams = Counter(generate_ngram(sentence, order))

    if not sentence_ngrams:
        return 0

    max_counts = {}
    for reference in references:
        ref_ngrams = Counter(generate_ngram(reference, order))
        for ngram in sentence_ngrams:
            max_counts[ngram] = max(max_counts.get(ngram, 0),
                                    ref_ngrams[ngram])

    # Intersection with clipping: limit by max reference count per n-gram
    clipped_counts = {
        ngram: min(count, max_counts.get(ngram, 0))
        for ngram, count in sentence_ngrams.items()
    }

    numerator = sum(clipped_counts.values())
    denominator = max(1, len(sentence) - order + 1)

    return numerator / denominator


def cumulative_bleu(references, sentence, n):
    """Compute cumulative BLEU up to order n.

    Args:
        references (list[list[str]]): Reference translations; tokenized.
        sentence (list[str]): Candidate translation; tokenized.
        n (int): Maximum n-gram order to include (1..n).

    Returns:
        float: Cumulative BLEU in [0, 1]. Uses geometric mean of precisions
            for orders 1..n and a brevity penalty. Returns 0 if any precision
            is 0 (no smoothing).
    """
    len_sentence = len(sentence)
    ngram_precisions = []
    for order in range(1, n + 1):
        ngram_precisions.append(modified_precision(
            references, sentence, order))

    # Brevity penalty using the closest reference length
    closest_ref_len = min((abs(len(ref) - len_sentence), len(ref))
                          for ref in references)[1]
    if len_sentence > closest_ref_len:
        BP = 1
    else:
        BP = np.exp(1 - closest_ref_len / len_sentence)

    # No smoothing: if any order has zero precision, BLEU is zero
    if any(p == 0 for p in ngram_precisions):
        return 0.0

    # Geometric mean of precisions
    bleu_score = BP * np.exp(sum(np.log(p) for p in ngram_precisions) / n)

    return bleu_score
