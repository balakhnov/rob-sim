import numpy as np


class CartPole:
    def __init__(self, params) -> None:
        self.nx = 4
        self.nu = 1

        self.gravity = 9.8
        self.masscart = params["masscart"]
        self.masspole = params["masspole"]
        self.total_mass = self.masspole + self.masscart
        self.length = params["length"]
        self.polemass_length = self.masspole * self.length
        self.force_mag = params["force_mag"]

        print("Pendulum created: ")
        for k, v in params.items():
            print(k, ":", v)
        print()

    def dynamic_equation(self, X, U):
        x = X[0]
        x_dot = X[1]
        theta = X[2]
        theta_dot = X[3]

        force = self.force_mag*U[0]

        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot ** 2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        dX = np.array([x_dot,xacc,theta_dot,thetaacc])
        return dX
