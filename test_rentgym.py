from unittest import TestCase
from rentgym import RentGym
import numpy as np


class TestRentGym(TestCase):
    def setUp(self):
        self.env = RentGym()


class TestInit(TestRentGym):
    def test_initial_apt(self):
        apt, p, r = self.env.reset()
        self.assertEqual(apt.omega.shape[0], self.env.n_omega)
        self.assertEqual(apt.upsilon.shape[0], self.env.n_upsilon)


# noinspection PyTypeChecker
class TestMus(TestRentGym):
    def setUp(self):
        self.env = RentGym(observables=400000, unobservables=400000)

    def test_mus(self):
        apt, p, r = self.env.reset()
        self.assertAlmostEqual(np.mean(p), 0.5, places=2)
        self.assertAlmostEqual(np.mean(r), 0.5, places=2)
        done = False
        omegas = []
        upsilons = []
        while not done:
            apt, reward, done, info = self.env.step(0)
            omegas.append(np.mean(apt.omega))
            upsilons.append(np.mean(apt.upsilon))
        self.assertAlmostEqual(np.mean(omegas + upsilons), 0.5, places=1)


class TestAction(TestRentGym):
    def test_action(self):
        apt, p, r = self.env.reset()
        reward = 0
        for eps in np.arange(0, 1, 0.01):
            act = np.random.choice([False, True], p=[1-eps, eps])
            apt, reward, done, _ = self.env.step(act)
            if done:
                break
        self.assertTrue(reward != 0)

    def test_timeout(self):
        done = False
        apt, p, r = self.env.reset()
        t = 0
        while not done:
            apt, reward, done, info = self.env.step(0)
            self.assertLess(t, 100)
