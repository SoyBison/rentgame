import torch
from torch.nn.functional import relu
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple
import numpy as np
from rent_dms import Strategy
import random
from rentgym import RentGym
import math
import matplotlib.pyplot as plt
from itertools import count

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def choose_action(self, p: np.ndarray, r: np.ndarray, omega: np.ndarray, c: int, mu_v=None):
        with torch.no_grad():
            decision = self.network({'p': p, 'r': r, 'oo': omega, 'c': c})
            return bool(decision)


class DQN(nn.Module):
    def __init__(self, n_omega, n_upsilon, preprocdim=32):
        super(DQN, self).__init__()
        # Note, I use 'oo' as a shorthand for omega.
        self.p_proc = nn.Linear(n_omega, preprocdim)
        self.r_proc = nn.Linear(n_upsilon, preprocdim)
        self.oo_proc = nn.Linear(n_omega, preprocdim)

        self.fc0 = nn.Linear(preprocdim * 3 + 1, 128)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

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
        print(p.shape)
        print(r.shape)
        print(oo.shape)
        print(x['c'].shape)
        h = torch.cat([p, r, oo, x['c']], 1)

        h = relu(self.fc0(h))
        h = relu(self.fc1(h))
        h = relu(self.fc2(h))
        y = torch.heaviside(self.fc3(h), torch.tensor([0], dtype=torch.double, device=device)).int()

        return y


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


def main():
    num_episodes = 1000
    env = RentGym()
    episode_durations = []

    policy_net = DQN(env.n_omega, env.n_upsilon).to(device)
    target_net = DQN(env.n_omega, env.n_upsilon).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    memory = ReplayMemory(10000)

    steps_done = 0

    def select_action(gamestate):
        nonlocal steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return policy_net(gamestate)
        else:
            return torch.tensor([[random.randint(0, 1)]], device=device, dtype=torch.int64)

    def optimize_model():
        if len(memory) < BATCH_SIZE:
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

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = target_net(non_final_next_states)

        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        loss = nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        policy_net.optim.zero_grad()
        loss.backward()

        policy_net.optim.step()

    rewards = []
    for i_ep in range(num_episodes):
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
                print(reward_tot)
                rewards.append(reward_tot)
                break

        if i_ep % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print('Complete')


if __name__ == '__main__':
    main()
