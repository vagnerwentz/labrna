import numpy as np

from labrna.perceptron import Perceptron

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y_E = np.array([0,
                0,
                0,
                1])
y_OU = np.array([0,
                 1,
                 1,
                 1])

perceptron_E = Perceptron(dimensionalidade=2)
perceptron_E.aprender(X, y_E)

perceptron_OU = Perceptron(dimensionalidade=2)
perceptron_OU.aprender(X, y_OU)

print("Predições E:", perceptron_E.predizer(X))
print("Predições OU:", perceptron_OU.predizer(X))
