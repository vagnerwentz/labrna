import numpy as np


def degrau(X):
    return np.heaviside(X, 0.5)


def concatena_1s(X):
    return np.hstack([X, np.ones((X.shape[0], 1))])  # Concatena uma coluna de 1s para multiplicar o "bias".


class Perceptron:
    def __init__(self, dimensionalidade, taxa_de_aprendizado=0.1, épocas=10, semente=0):
        rnd = np.random.default_rng(semente)
        self.W = rnd.normal(size=dimensionalidade + 1)  # A matriz de pesos inclui peso 1 para o intercepto ("bias").
        self.λ = taxa_de_aprendizado
        self.épocas = épocas

    def predizer(self, X):
        X = concatena_1s(X)
        return degrau(X @ self.W)

    def aprender(self, X, Y):
        X = concatena_1s(X)
        for época in range(self.épocas):
            # Atualiza matriz de pesos a cada nova instância.
            for x, y in zip(X, Y):
                predição = degrau(x @ self.W)
                erro = y - predição  # A variável 'erro' dá a direção da correção.
                self.W += self.λ * erro * x
