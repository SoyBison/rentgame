import torch
from tqdm import tqdm
from torch.nn.functional import relu
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple
import numpy as np
from rent_dms import Strategy, Rule37, RandomActor, ValueThreshold
import random
from rentgym import RentGym
import math
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import count

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

device = torch.device("cuda")

TRAIN_TIME = 5000
BATCH_SIZE = 100
GAMMA = 0.999
EPS_START = 0.99
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


class TorchStrat(Strategy):
    """
    Set up a pre-trained pytorch network to evaluate. Do not use this object for training.
    """

    def __init__(self, network):
        super(TorchStrat, self).__init__()
        self.network = network
        self.network.eval()
        if hasattr(self.network, 'memory'):
            self.memories = self.network.memory.cpu()

    def choose_action(self, p: np.ndarray, r: np.ndarray, omega: np.ndarray, c: int, mu_v=None):
        super(TorchStrat, self).choose_action(p, r, omega, c, mu_v)
        with torch.no_grad():
            p = torch.tensor([p], device=device, dtype=torch.double)
            r = torch.tensor([r], device=device, dtype=torch.double)
            omega = torch.tensor([omega], device=device, dtype=torch.double)
            c = torch.tensor([[c]], device=device, dtype=torch.double)

            decision = self.network({'p': p, 'r': r, 'oo': omega, 'c': c})
            if hasattr(self.network, 'memory'):
                self.memories = torch.vstack([self.memories, self.network.memory.cpu()])
            return decision.max(1)[1].item()


class DQN(nn.Module):
    def __init__(self, n_omega, n_upsilon, preprocdim=32):
        super(DQN, self).__init__()
        # Note, I use 'oo' as a shorthand for omega.
        self.p_proc = nn.Linear(n_omega, preprocdim)
        self.r_proc = nn.Linear(n_upsilon, preprocdim)
        self.oo_proc = nn.Linear(n_omega, preprocdim)

        self.fc0 = nn.Linear(preprocdim * 3 + 1, 128)
        self.head = nn.Linear(128, 2)

        self.optim = optim.Adam(self.parameters(), lr=0.001)
        self.double()

    def forward(self, x):
        """
        Make sure that x is a dict with p, r, omega, and c.
        :param x: dict
        :return: bool
        """
        p = relu(self.p_proc(x['p']))
        r = relu(self.r_proc(x['r']))
        oo = relu(self.oo_proc(x['oo']))
        h = torch.cat([p, r, oo, x['c']], 1)

        h = relu(self.fc0(h))

        y = self.head(h)

        return y


class DQNWMT(nn.Module):
    def __init__(self, n_omega, n_upsilon, preprocdim=32, memory_size=32):
        super(DQNWMT, self).__init__()
        self.memory_size = memory_size
        self.p_proc = nn.Linear(n_omega, preprocdim)
        self.r_proc = nn.Linear(n_upsilon, preprocdim)
        self.oo_proc = nn.Linear(n_omega, preprocdim)

        self.memory = torch.zeros((1, memory_size), device=device)

        self.fc0 = nn.Linear(preprocdim * 3 + 1 + memory_size, 128)

        self.head = nn.Linear(128, 2)

        self.memorizer = nn.Linear(128, memory_size)

        self.optim = optim.Adam(self.parameters(), lr=0.0025)
        self.double()

    def forward(self, x, opti=False):
        p = relu(self.p_proc(x['p']))
        r = relu(self.r_proc(x['r']))
        oo = relu(self.oo_proc(x['oo']))
        if opti:
            h = torch.cat([p, r, oo, x['c'], x['mem']], 1)
        else:
            h = torch.cat([p, r, oo, x['c'], self.memory], 1)

        h = relu(self.fc0(h))

        y = self.head(h)
        if not opti:
            self.memory = self.memorizer(h)

        return y

    def eternal_sunshine(self):
        self.memory = torch.zeros((1, self.memory_size), device=device)


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def plot_durations(durations):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(durations, dtype=torch.double)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())

    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.show()


def train_memtens():
    num_episodes = TRAIN_TIME
    env = RentGym()
    episode_durations = []
    policy_net = DQNWMT(env.n_omega, env.n_upsilon).to(device)
    target_net = DQNWMT(env.n_omega, env.n_upsilon).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    memory = ReplayMemory(100000)

    steps_done = 0

    def select_action(gamestate):
        nonlocal steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return policy_net(gamestate).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[0]], device=device, dtype=torch.int64)

    def optimize_model():
        if len(memory) <= BATCH_SIZE:
            return
        transition = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transition))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = {
            'p': torch.vstack([s['p'] for s in batch.next_state if s is not None]),
            'r': torch.vstack([s['r'] for s in batch.next_state if s is not None]),
            'oo': torch.vstack([s['oo'] for s in batch.next_state if s is not None]),
            'c': torch.vstack([s['c'] for s in batch.next_state if s is not None]),
            'mem': torch.vstack([s['mem'] for s in batch.next_state if s is not None])
        }

        state_batch = {
            'p': torch.vstack([s['p'] for s in batch.state if s is not None]),
            'r': torch.vstack([s['r'] for s in batch.state if s is not None]),
            'oo': torch.vstack([s['oo'] for s in batch.state if s is not None]),
            'c': torch.vstack([s['c'] for s in batch.state if s is not None]),
            'mem': torch.vstack([s['mem'] for s in batch.state if s is not None])
        }
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = policy_net.forward(state_batch, opti=True).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device, dtype=torch.double)
        action_batch = action_batch[non_final_mask]
        temp = target_net.forward(non_final_next_states, opti=True).gather(1, action_batch)
        next_state_values[non_final_mask] = temp.view(1, -1)

        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        loss = nn.functional.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        policy_net.optim.zero_grad()
        loss.backward()

        policy_net.optim.step()

    rewards = []
    pbar = tqdm(range(num_episodes))
    for i_ep in pbar:
        apt, p, r = env.reset()
        state = {'p': torch.tensor([p], device=device),
                 'r': torch.tensor([r], device=device),
                 'oo': torch.tensor([apt.omega], device=device, dtype=torch.double),
                 'c': torch.tensor([[apt.c]], device=device),
                 'mem': torch.zeros_like(policy_net.memory, device=device)}
        reward_tot = 0
        for t in count():
            action = select_action(state)
            apt, reward, done, _ = env.step(action)
            reward_tot += reward
            reward = torch.tensor([reward], device=device, dtype=torch.double)
            if not done:
                next_state = {'p': torch.tensor([p], device=device),
                              'r': torch.tensor([r], device=device),
                              'oo': torch.tensor([apt.omega], device=device, dtype=torch.double),
                              'c': torch.tensor([[apt.c]], device=device),
                              'mem': policy_net.memory}
            else:
                next_state = None

            memory.push(state, action, next_state, reward)

            state = next_state

            optimize_model()
            if done:
                episode_durations.append(t + 1)
                rewards.append(reward_tot)
                break

        if i_ep % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            pbar.set_description(f'Score (50 ep avg): {np.mean(rewards[-50:]):0.5}')
    tqdm.write('Training Complete')
    return policy_net


def train_dqn():
    num_episodes = TRAIN_TIME
    env = RentGym()
    episode_durations = []

    policy_net = DQN(env.n_omega, env.n_upsilon).to(device)
    target_net = DQN(env.n_omega, env.n_upsilon).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    memory = ReplayMemory(100000)

    steps_done = 0

    def select_action(gamestate):
        nonlocal steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return policy_net(gamestate).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randint(0, 1)]], device=device, dtype=torch.int64)

    def optimize_model():
        if len(memory) <= BATCH_SIZE:
            return
        transition = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transition))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = {
            'p': torch.vstack([s['p'] for s in batch.next_state if s is not None]),
            'r': torch.vstack([s['r'] for s in batch.next_state if s is not None]),
            'oo': torch.vstack([s['oo'] for s in batch.next_state if s is not None]),
            'c': torch.vstack([s['c'] for s in batch.next_state if s is not None]),
        }

        state_batch = {
            'p': torch.vstack([s['p'] for s in batch.state if s is not None]),
            'r': torch.vstack([s['r'] for s in batch.state if s is not None]),
            'oo': torch.vstack([s['oo'] for s in batch.state if s is not None]),
            'c': torch.vstack([s['c'] for s in batch.state if s is not None]),
        }
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device, dtype=torch.double)
        action_batch = action_batch[non_final_mask]
        temp = target_net(non_final_next_states).gather(1, action_batch)
        next_state_values[non_final_mask] = temp.view(1, -1)

        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        loss = nn.functional.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        policy_net.optim.zero_grad()
        loss.backward()

        policy_net.optim.step()

    rewards = []
    pbar = tqdm(range(num_episodes))
    for i_ep in pbar:
        apt, p, r = env.reset()
        state = {'p': torch.tensor([p], device=device),
                 'r': torch.tensor([r], device=device),
                 'oo': torch.tensor([apt.omega], device=device, dtype=torch.double),
                 'c': torch.tensor([[apt.c]], device=device)}
        reward_tot = 0
        for t in count():
            action = select_action(state)
            apt, reward, done, _ = env.step(action)
            reward_tot += reward
            reward = torch.tensor([reward], device=device, dtype=torch.double)
            if not done:
                next_state = {'p': torch.tensor([p], device=device),
                              'r': torch.tensor([r], device=device),
                              'oo': torch.tensor([apt.omega], device=device, dtype=torch.double),
                              'c': torch.tensor([[apt.c]], device=device)}
            else:
                next_state = None

            memory.push(state, action, next_state, reward)

            state = next_state

            optimize_model()
            if done:
                episode_durations.append(t + 1)
                rewards.append(reward_tot)
                break

        if i_ep % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            pbar.set_description(f'Score (50 ep avg): {np.mean(rewards[-50:]):0.5}')
    tqdm.write('Training Complete')
    return policy_net


class PiThreshold:
    """Created to test the hypothesis that the deep-q network is just finding a profit threshold that works for all
    games, leveraging the statistical properties. This'll use a simple binary threshold. This is an extremely simple
    reinforcement learner."""

    def __init__(self, eta=.01):
        self.pi = np.random.randint(500, 2000)
        self.eta = eta

    def update(self, score):
        if score == 0:
            self.pi -= self.eta * self.pi
        elif score < 0:
            self.pi -= score * self.eta

    def train(self):
        while True:
            lastpi = self.pi
            strat = ValueThreshold(self.pi)
            score = strat.play()
            self.update(score)
            if score <= 0 and np.isclose(self.pi, lastpi):
                break


def main():
    tqdm.write('Training Memory Tensor Model')
    memtenspnet = train_memtens()
    tqdm.write('Training Deep Q Model')
    policy_net = train_dqn()
    pithresh = PiThreshold()
    tqdm.write('Training Threshold Model')
    pithresh.train()
    thrstrat = ValueThreshold(threshold=pithresh.pi)
    tqdm.write('Testing Threshold Model')
    thrscores = thrstrat.play_n()
    dqnstrat = TorchStrat(policy_net)
    tqdm.write('Testing Deep Q Model')
    dqnscores = dqnstrat.play_n()
    mtstrat = TorchStrat(memtenspnet)
    tqdm.write('Testing Memory Tensor Model')
    mtscores = mtstrat.play_n()
    r37strat = Rule37()
    tqdm.write('Testing Rule 37')
    r37scores = r37strat.play_n()
    randstrat = RandomActor()
    tqdm.write('Testing Random Actor')
    randscores = randstrat.play_n()

    sns.kdeplot(dqnscores, label='Deep Q')
    sns.kdeplot(r37scores, label='37 Rule')
    sns.kdeplot(randscores, label='Random Actor')
    sns.kdeplot(thrscores, label='Threshold Learning')
    sns.kdeplot(mtscores, label='Memory Tensor Deep Q')

    plt.legend()
    plt.title('Distribution of Scores')
    plt.show()

    sns.kdeplot(dqnstrat.buy_ts, label='Deep Q', clip=(0, 100))
    sns.kdeplot(r37strat.buy_ts, label='37 Rule', clip=(0, 100))
    sns.kdeplot(randstrat.buy_ts, label='Random Actor', clip=(0, 100))
    sns.kdeplot(thrstrat.buy_ts, label='Threshold Learning', clip=(0, 100))
    sns.kdeplot(mtstrat.buy_ts, label='Memory Tensor Deep Q', clip=(0, 100))

    plt.legend()
    plt.title('Distribution of Buy Periods')
    plt.show()

    freshstrat = TorchStrat(memtenspnet)
    freshstrat.play_n(n=10)
    plt.imshow(freshstrat.memories)
    plt.title('Memory Tensor over Time in single test game.')
    plt.ylabel('Time')
    plt.show()

    tqdm.write(f'In Tests, Deep Q Scores {np.mean(dqnscores)} on Average.')
    tqdm.write(f'In Tests, the 37 Rule Scores {np.mean(r37scores)} on Average.')
    tqdm.write(f'In Tests, a Random Actor Scores {np.mean(randscores)} on Average.')
    tqdm.write(f'In Tests, The Memory Tensor Model Scores {np.mean(mtscores)} on Average.')

    tqdm.write(f'In Tests, Threshold Learning Scores {np.mean(thrscores)} on Average.')
    tqdm.write(f"The Threshold Learner's decision threshold is {pithresh.pi}.")


if __name__ == '__main__':
    main()
