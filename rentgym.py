import numpy as np


class RentGym:
    def __init__(self, periods=100, observables=16, unobservables=8):
        self.periods = periods
        self.action_space = np.array([False, True])
        self.n_omega = observables
        self.n_upsilon = unobservables
        self.apt = None
        self.done = False
        self.t = 0

        self.mu_c = np.random.randint(500, 2000)
        self.sigma_c = np.random.randint(100, 500)

        self.p = np.random.uniform(0, 1, observables)
        self.r = np.random.uniform(0, 1, unobservables)

        self.mu_v = np.random.randint(500, 2000)
        self.mu_value = (4 * self.mu_v) / (observables + unobservables)

    def reset(self):
        self.done = False
        self.apt = Apartment(self.n_omega, self.n_upsilon, self.mu_c, self.sigma_c)
        self.t = 0
        return self.apt, self.p, self.r

    def step(self, action):
        if self.done or self.t >= self.periods:
            self.done = True
            return self.apt, 0, True, {}
        elif action:
            value = self.mu_value * (self.p.dot(self.apt.omega) + self.r.dot(self.apt.upsilon))
            self.done = True
            return self.apt, value - self.apt.c, True, {}
        else:
            self.apt = Apartment(self.n_omega, self.n_upsilon, self.mu_c, self.sigma_c)
            self.t += 1
            return self.apt, 0, False, {}


class Apartment:
    def __init__(self, n_omega, n_upsilon, mu_c, sigma_c):
        self.n_omega = n_omega
        self.n_upsilon = n_upsilon
        self.mu_c = mu_c
        self.sigma_c = sigma_c

        while True:
            self.c = np.round(np.random.normal(self.mu_c, self.sigma_c))
            if 0 < self.c:
                break

        z = (self.c - self.mu_c) / self.sigma_c
        quality = (1 + z / (1 + np.abs(z))) * 0.5
        self.omega = np.random.choice([False, True], p=(1 - quality, quality), size=self.n_omega)
        self.upsilon = np.random.choice([False, True], p=(1 - quality, quality), size=self.n_upsilon)
