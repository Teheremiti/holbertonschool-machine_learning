#!/usr/bin/env python3
"""Interactive question answering on a single reference text."""

question_answer = __import__("0-qa").question_answer

QUIT_LIST = ["exit", "quit", "goodbye", "bye"]


def answer_loop(reference):
    """Interactively answer questions from a reference text.

    The function repeatedly prompts the user for a question using the
    prefix ``Q: ``. It computes an answer using :func:`question_answer`.
    If no answer is found, a default apology message is used instead.
    Typing any word in ``QUIT_LIST`` (case-insensitive) prints
    ``A: Goodbye`` and terminates the process.

    Args:
        reference (str): Reference text used to answer all questions.
    """
    while True:
        question = input("Q: ")
        if question.lower() in QUIT_LIST:
            print("A: Goodbye")
            exit()
        answer = question_answer(question, reference)
        if answer is None:
            answer = "Sorry, I do not understand your question."
        print(f"A: {answer}")
