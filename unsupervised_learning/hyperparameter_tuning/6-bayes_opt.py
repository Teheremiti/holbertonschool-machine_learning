#!/usr/bin/env python3

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from GPyOpt.methods import BayesianOptimization
import matplotlib.pyplot as plt
import joblib

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=0)


def objective_function(params):
    """
    Objective function for Bayesian optimization.
    """
    lr, hidden, dropout, alpha, batch = params[0]
    hidden = int(hidden)
    clf = MLPClassifier(
        hidden_layer_sizes=(hidden,),
        alpha=alpha,
        batch_size=int(batch),
        learning_rate_init=lr,
        max_iter=200,
        early_stopping=True,
        random_state=0
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_val)
    acc = accuracy_score(y_val, preds)
    # Save checkpoint only if validation improves
    global_best = [-np.inf]
    acc = accuracy_score(y_val, preds)
    if acc > global_best[0]:
        global_best[0] = acc
        fname = f"checkpoint_lr{lr:.5f}_h{hidden}_d{dropout:.2f}_a{alpha:.5f}_b{int(batch)}.pkl"
        joblib.dump(clf, fname)
    return -acc


if __name__ == '__main__':

    # Define bounds
    domain = [
        {'name': 'lr', 'type': 'continuous', 'domain': (1e-5, 1e-1)},
        {'name': 'hidden', 'type': 'discrete', 'domain': (10, 50, 100, 200)},
        {'name': 'dropout', 'type': 'continuous', 'domain': (0.0, 0.5)},
        {'name': 'alpha', 'type': 'continuous', 'domain': (1e-6, 1e-2)},
        {'name': 'batch', 'type': 'discrete', 'domain': (16, 32, 64, 128)}
    ]

    # Setup and run Bayesian optimization
    bo = BayesianOptimization(
        f=objective_function,
        domain=domain,
        model_type='GP',
        acquisition_type='EI',
        exact_feval=True,
        maximize=False
    )

    max_iter = 50
    bo.run_optimization(max_iter=max_iter)

    # Optimal hyperparameters
    best_idx = np.argmin(bo.Y)
    x_opt = bo.X[best_idx]
    y_opt = bo.Y[best_idx]
    print("Optimal hyperparameters:", x_opt)
    print("Validation accuracy:", -y_opt)

    # Convergence plot
    plt.figure()
    plt.plot(-bo.Y, marker='o')
    plt.title("Bayesian Optimization Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Validation Accuracy")
    plt.savefig('convergence.png')
    plt.show()

    # Save report
    with open('bayes_opt.txt', 'w') as f:
        f.write("Optimization Report\n")
        f.write("====================\n")
        f.write("Best hyperparameters:\n")
        f.write(str(x_opt) + '\n')
        f.write("Best validation accuracy: {:.4f}\n".format(-float(y_opt)))
        f.write("\nAll evaluated scores:\n")
        f.write(str(-bo.Y.reshape(-1)) + '\n')
