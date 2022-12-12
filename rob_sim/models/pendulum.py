import numpy as np


class Pendulum:
    def __init__(self, params) -> None:
        self.nx = 2
        self.nu = 1
        
        self.m = params["m"]
        self.l = params["l"]
        self.k_i = params["k_i"]
        self.k = params["k"]
        self.g = params["g"]

        self.I = self.m*self.l**2

        print("Pendulum created: ")
        for k, v in params.items():
            print(k, ":", v)
        print()

    def dynamic_equation(self, X, U):
        dX = np.array([X[1], (self.k_i * U[0] + self.m * self.g *
                      self.l * np.sin(X[0]) - self.k * X[1]) / self.I])
        return dX
