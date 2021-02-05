from abc import abstractmethod

from rentgym import RentGym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class Strategy:
    def __init__(self):
        self.t = 0
        self.ts = []
        self.network = None
        pass

    @abstractmethod
    def choose_action(self, p: np.ndarray, r: np.ndarray, omega: np.ndarray, c: int, mu_v):
        """
        Takes in a full observation and returns a decision on whether or not to buy.
        :param mu_v: int
        :param p: np.ndarray
        :param r: np.ndarray
        :param omega: np.ndarray
        :param c: int
        :return: bool
        """
        self.t += 1
        pass

    def play(self, **kwargs):
        env = RentGym(**kwargs)
        apt, p, r = env.reset()
        # try:
        #     self.network.eternal_sunshine()
        # except AttributeError:
        #     pass
        while True:
            apt, reward, done, _ = env.step(self.choose_action(p, r, apt.omega, apt.c, env.mu_v))
            if done:
                self.ts.append(self.t)
                self.t = 0
                break
        return reward

    def play_n(self, n=10000, **kwargs):
        rewards = []
        for _ in tqdm(range(n)):
            rewards.append(self.play(**kwargs))

        return rewards

    @property
    def buy_ts(self):
        return self.ts


class RandomActor(Strategy):
    def __init__(self, p_buy=0.075):
        super(RandomActor, self).__init__()
        self.p_buy = p_buy

    def choose_action(self, p: np.ndarray, r: np.ndarray, omega: np.ndarray, c: int, mu_v):
        super(RandomActor, self).choose_action(p, r, omega, c, mu_v)
        return np.random.choice([True, False], p=[self.p_buy, 1 - self.p_buy])


class ValueThreshold(Strategy):
    def __init__(self, threshold):
        super(ValueThreshold, self).__init__()
        self.threshold = threshold
        self.obs_score = None

    def choose_action(self, p: np.ndarray, r: np.ndarray, omega: np.ndarray, c: int, mu_v):
        super(ValueThreshold, self).choose_action(p, r, omega, c, mu_v)
        obs_score = (4 * mu_v * p.dot(omega) / len(p)) - c
        self.obs_score = obs_score
        if obs_score >= self.threshold:
            return 1
        else:
            return 0


class Rule37(Strategy):
    def __init__(self):
        super(Rule37, self).__init__()
        self.t = 0
        self.t_switch = 37
        self.memory = []

    def choose_action(self, p: np.ndarray, r: np.ndarray, omega: np.ndarray, c: int, mu_v):
        super(Rule37, self).choose_action(p, r, omega, c, mu_v)

        obs_score = (4 * mu_v * p.dot(omega) / len(p)) - c
        if self.t < self.t_switch:
            self.memory.append(obs_score)
            return 0
        elif np.all(np.greater(obs_score, self.memory)):
            return 1
        else:
            self.memory.append(obs_score)
            return 0

    def play(self, **kwargs):
        self.t = 0
        if 'periods' in kwargs:
            self.t_switch = np.round((1 / np.e) * kwargs['periods'])
        else:
            self.t_switch = 37
        self.memory = []
        return super(Rule37, self).play(**kwargs)


if __name__ == '__main__':
    rule37 = Rule37()
    r37scores = rule37.play_n(n=5000)
    rand = RandomActor()
    randscores = rand.play_n(n=100000)
    sns.kdeplot(r37scores, label='Rule 37')
    sns.kdeplot(randscores, label='Random Actor')
    plt.title('Score Distributions')
    plt.legend()
    plt.show()
    sns.kdeplot(rule37.buy_ts)
    plt.title('Time at which rule37 buys')
    plt.show()
    print(f'Rule 37 On Average Scores {np.mean(r37scores)}')
    print(f'A Random Actor On Average Scores {np.mean(randscores)}')
