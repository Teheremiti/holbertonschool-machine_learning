#!/usr/bin/env python3
""" Stacking Bars """

import numpy as np
import matplotlib.pyplot as plt


def bars():
    """ Plots a stacked bar graph. """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    names = ['Farrah', 'Fred', 'Felicia']
    colors = ['r', 'yellow', '#ff8000', '#ffe5b4']
    labels = ['apples', 'bananas', 'oranges', 'peaches']

    bottom = 0
    for i in range(len(fruit)):
        plt.bar(
            names,
            fruit[i],
            width=0.5,
            bottom=bottom,
            color=colors[i],
            label=labels[i]
            )
        bottom += fruit[i]

    plt.title('Number of Fruit per Person')
    plt.ylabel('Quantity of Fruit')
    plt.yticks(range(0, 81, 10))
    plt.legend()

    plt.show()
