#!/usr/bin/env python3
"""Interactive loop that prompts for questions and prints responses."""

quit_list = ["exit", "quit", "goodbye", "bye"]


def loop_QaBot():
    """Run an interactive question loop.

    The function repeatedly prompts the user for a question on standard
    input using the prefix ``Q: ``. If the user types a word contained in
    ``quit_list`` (case-insensitive), the function prints ``A: Goodbye``
    and terminates the process. Otherwise, it prints a placeholder
    response prefix ``A: ``.
    """
    while True:
        question = input("Q: ")
        if question.lower() in quit_list:
            print("A: Goodbye")
            exit()
        print("A: ")


if __name__ == "__main__":
    loop_QaBot()
