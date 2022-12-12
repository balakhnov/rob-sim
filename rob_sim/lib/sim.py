import numpy as np

class ModelFactory:
    def __init__(self, model, params) -> None:
        self.model = model
        self.params = params

    def create(self):
        concret_params = {k: v.sample() for k, v in self.params.items()}
        return self.model(concret_params)


def RK45(dynamic_equation, X, U, dt):
    k1 = dynamic_equation(X, U)
    k2 = dynamic_equation(X + dt / 2 * k1, U)
    k3 = dynamic_equation(X + dt / 2 * k2, U)
    k4 = dynamic_equation(X + dt * k3, U)
    return X + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

class Logger:
    def __init__(self, X0 = None) -> None:
        self.log_state = []
        self.log_action = []

    def add_step(self,X,U):
        self.log_state.append(X)
        self.log_action.append(U)

    def reset(self, X0):
        self.log_state = []
        self.log_action = []
        self.log_state.append(X0)

    def get_log_state(self):
        return np.array(self.log_state).T

    def get_log_action(self):
        return np.array(self.log_action).T

class Simulator:
    def __init__(self, model, dt, meas_noise, X0, logger = None) -> None:
        self.model = model
        self.dt = dt
        self.X = X0
        self.meas_noise = meas_noise

        self.logger = logger
        if self.logger:
            self.logger.reset(X0)


    def step(self, U):
        self.X = RK45(self.model.dynamic_equation, self.X, U, self.dt.sample())

        if self.logger:
            self.logger.add_step(self.X,U)

    def get_state(self):
        return self.X

    def get_observe(self):
        return self.X + self.meas_noise.sample()

    def reset(self, X0):
        self.X = X0

        if self.logger:
            self.logger.reset(X0)

    def get_log_state(self):
        if self.logger:
            return self.logger.get_log_state()
        else:
            # print("logger doesn't initialized")
            raise RuntimeError("logger doesn't initialized")

    def get_log_action(self):
        if self.logger:
            return self.logger.get_log_action()
        else:
            # print("logger doesn't initialized")
            raise RuntimeError("logger doesn't initialized")
