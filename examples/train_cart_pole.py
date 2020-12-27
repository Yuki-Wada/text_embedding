"""
Train a Q-Value.
"""
# state:
#       cart's position: -2.4 - 2.4
#       cart's velocity: -3.0 - 3.0
#       bar's angle (radian):   -2.4 - 2.4
#       bar's angular velocity(radian): -2.0 - 2.0
# termination:
#       cart's position > 2.4 or cart's position< -2.4
#       bar's angle > 15 / 2 / pi or bar's angle < 15 / 2 / pi

import os
import argparse
import logging
from itertools import product
from queue import Queue
import numpy as np
import gym
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mltools.utils import set_seed, set_logger, dump_json

logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", default='data/model/cart_pole/')
    parser.add_argument('--time_step_plot', default='time_step.png')

    parser.add_argument("--episode_count", type=int, default=2000)

    parser.add_argument("--algorithm", default='montecarlo')
    parser.add_argument("--n_step", type=int, default=1)
    parser.add_argument("--first_visit", action='store_true')

    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.1)

    parser.add_argument("--render", action='store_true')

    parser.add_argument("--seed", type=int)

    args = parser.parse_args()

    return args

def plot(values, label, figure_path):
    episode_numbers = np.arange(0, len(values))
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(episode_numbers, np.array(values)[episode_numbers], label='time_step')
    ax.set_xlabel('Episode')
    ax.set_ylabel(label)
    ax.legend()

    x_ax = ax.get_xaxis()
    x_ax.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(figure_path)
    plt.close()

class CartPoleStateConverter:
    def __init__(self, epsilon=0.2):
        self.epsilon = epsilon

        self.min_position = -2.4
        self.max_position = 2.4
        self.position_size = 6

        self.min_velocity = -2.0
        self.max_velocity = 2.0
        self.velocity_size = 6

        self.min_angle = -0.5
        self.max_angle = 0.5
        self.angle_size = 6

        self.min_angular_velocity = -1.5
        self.max_angular_velocity = 1.5
        self.angular_velocity_size = 6

        self.action_num = 2

    def state_to_index(self, state):
        def bins(clip_min, clip_max, num):
            return np.linspace(clip_min, clip_max, num + 1)[1:-1]

        position, velocity, angle, angular_velocity = state

        position_index = np.digitize(position, bins=bins(
            self.min_position, self.max_position, self.position_size))
        velocity_index = np.digitize(velocity, bins=bins(
            self.min_velocity, self.max_velocity, self.velocity_size))
        angle_index = np.digitize(angle, bins=bins(
            self.min_angle, self.max_angle, self.angle_size))
        angular_velocity_index = np.digitize(angular_velocity, bins=bins(
            self.min_angular_velocity, self.max_angular_velocity, self.angular_velocity_size))
        
        return position_index, velocity_index, angle_index, angular_velocity_index

    def get_initial_q_value(self):
        q_value = np.random.uniform(low=0, high=1, size=(
            self.position_size,
            self.velocity_size,
            self.angle_size,
            self.angular_velocity_size,
            self.action_num,
        ))
        q_value[0] = 0
        q_value[-1] = 0
        q_value[:, :, 0] = 0
        q_value[:, :, -1] = 0

        return q_value

    def state_to_index2(self, state):
        def bins(clip_min, clip_max, num):
            return np.linspace(clip_min, clip_max, num + 1)[1:-1]

        position, velocity, angle, angular_velocity = state

        position_index = np.digitize(position, bins=bins(
            self.min_position, self.max_position, self.position_size))
        velocity_index = np.digitize(velocity, bins=bins(
            self.min_velocity, self.max_velocity, self.velocity_size))
        angle_index = np.digitize(angle, bins=bins(
            self.min_angle, self.max_angle, self.angle_size))
        angular_velocity_index = np.digitize(angular_velocity, bins=bins(
            self.min_angular_velocity, self.max_angular_velocity, self.angular_velocity_size))
        
        return position_index + velocity_index * 6 + angle_index * 36 + angular_velocity_index * 216

    def get_initial_v_value(self):
        v_value = np.random.uniform(low=0, high=1, size=(
            self.position_size,
            self.velocity_size,
            self.angle_size,
            self.angular_velocity_size,
        ))
        v_value[0] = 0
        v_value[-1] = 0
        v_value[:, :, 0] = 0
        v_value[:, :, -1] = 0

        v_value = v_value.reshape(-1)

        return v_value

def get_action(q_value, state_converter, state):
    index = state_converter.state_to_index(state)
    if np.random.uniform() <= state_converter.epsilon:
        return np.random.randint(0, state_converter.action_num)
    return np.argmax(q_value[index])

def setup_output_dir(output_dir_path, args):
    os.makedirs(output_dir_path, exist_ok=True)
    dump_json(args, os.path.join(output_dir_path, 'args.json'))

def get_transition_prob(env, state_converter):
    action = 0
    appearances = np.random.uniform(low=0, high=1, size=(
        6 * 6 * 6 * 6,
        2,
        6 * 6 * 6 * 6,
        2,
    ))
    for _ in range(100000 * 2):
        curr_state = env.reset()
        action = 1 - action
        next_state, reward, is_terminated, _ = env.step(action)
        if is_terminated:
            reward_index = 1
        else:
            reward_index = 0
        
        curr_state_index = state_converter.state_to_index2(curr_state)
        next_state_index = state_converter.state_to_index2(next_state)

        appearances[next_state_index][reward_index][curr_state_index][action] += 1

    transition_prob = np.empty_like(appearances)
    for state in range(1296):
        transition_prob[state][0] = appearances[state][0] / (np.sum(appearances[state][0]) + 1e-18)
        transition_prob[state][1] = appearances[state][1] / (np.sum(appearances[state][1]) + 1e-18)

    return transition_prob

def value_iteration(
        env,
        state_converter,
        alpha=0.1,
        gamma=0.95,
        episode_count=2000,
        render=False,
        figure_path=None,
    ):
    def update_v_value(v_value, transition_prob):
        rewards = np.expand_dims(np.array([1, -10]), axis=(1, 2))

        v_value = np.max(
            np.sum(
                transition_prob * (rewards + np.expand_dims(v_value, axis=(0, 2))), axis=(0, 1),
            ), axis=-1
        )

    transition_prob = get_transition_prob(env, state_converter)
    v_value = state_converter.get_initial_v_value()

    time_steps = []
    for i in range(episode_count):
        update_v_value(v_value, transition_prob)

        prev_state = env.reset()
        for t in range(1000):
            if render:
                env.render()

            vs = []
            for action in range(2):
                state, _, _, _ = env.step(action)
                state_index = state_converter.state_to_index2(state)
                vs.append(v_value[state_index])
                env.state = prev_state
            action = np.argmax(vs)

            state, _, is_terminated, _ = env.step(action)

            prev_state = state
            if is_terminated:
                print("Episode {} finished after {} timesteps".format(i, t + 1))
                time_steps.append(t + 1)
                break

        if (i + 1) % 20 == 0:
            plot(time_steps, 'Time Step', figure_path)

def monte_carlo(
        env,
        state_converter,
        first_visit=False,
        gamma=0.95,
        episode_count=2000,
        render=False,
        figure_path=None,
    ):
    q_value = state_converter.get_initial_q_value()
    appearances = np.zeros_like(q_value)

    def update_q_value(q_value, state, action, G):
        index = state_converter.state_to_index(state)

        q_value[index][action] = \
            (q_value[index][action] * appearances[index][action] + G) / (appearances[index][action] + 1)
        appearances[index][action] += 1

    time_steps = []
    for i in range(episode_count):
        quadruplet = []
        prev_state = env.reset()
        seen_in_episode = set()
        for t in range(1000):
            if render:
                env.render()

            action = get_action(q_value, state_converter, prev_state)
            state, reward, is_terminated, _ = env.step(action)
            reward = -10 if is_terminated and t < 195 else 1

            idx = state_converter.state_to_index(state)
            quadruplet.append((prev_state, action, reward, (idx, action) not in seen_in_episode))
            seen_in_episode.add((idx, action))

            prev_state = state
            if is_terminated:
                print("Episode {} finished after {} timesteps".format(i, t + 1))
                time_steps.append(t + 1)
                break

        G = 0
        for state, action, reward, use in quadruplet[::-1]:
            G = G * gamma + reward
            if use:
                update_q_value(q_value, state, action, G)

        if (i + 1) % 20 == 0:
            plot(time_steps, 'Time Step', figure_path)

def sarsa(
        env,
        state_converter,
        n_step=1,
        alpha=0.1,
        gamma=0.95,
        episode_count=2000,
        render=False,
        figure_path=None,
    ):
    def update_q_value(q_value, prev_state, prev_action, state, action, next_reward):
        prev_index = state_converter.state_to_index(prev_state)
        index = state_converter.state_to_index(state)

        q_value[prev_index][prev_action] = \
            (1 - alpha) * q_value[prev_index][prev_action] + \
            alpha * (next_reward + gamma * q_value[index][action])

    q_value = state_converter.get_initial_q_value()

    queue = Queue()
    gamma_n = gamma ** n_step

    time_steps = []
    for i in range(episode_count):
        if not queue.empty():
            queue.get_nowait()
        for _ in range(n_step):
            queue.put(0)

        G = 0
        prev_state = env.reset()
        prev_action = None
        for t in range(1000):
            if render:
                env.render()

            action = get_action(q_value, state_converter, prev_state)
            state, reward, is_terminated, _ = env.step(action)
            reward = -100 if is_terminated and t < 195 else 1 

            if prev_action is not None:
                update_q_value(q_value, prev_state, prev_action, state, action, reward)

            prev_state = state
            prev_action = action
            if is_terminated:
                print("Episode {} finished after {} timesteps".format(i, t + 1))
                time_steps.append(t + 1)
                break

        if (i + 1) % 20 == 0:
            plot(time_steps, 'Time Step', figure_path)

def q_learning(
        env,
        state_converter,
        n_step=1,
        alpha=0.1,
        gamma=0.95,
        episode_count=2000,
        render=False,
        figure_path=None,
    ):
    def update_q_value(q_value, curr_state, curr_action, next_state, next_reward):
        curr_index = state_converter.state_to_index(curr_state)
        next_index = state_converter.state_to_index(next_state)
        q_value[curr_index][curr_action] = \
            (1 - alpha) * q_value[curr_index][curr_action] + \
            alpha * (next_reward + gamma * np.max(q_value[next_index]))

    q_value = state_converter.get_initial_q_value()

    queue = Queue()
    gamma_n = gamma ** n_step

    time_steps = []
    for i in range(episode_count):
        if not queue.empty():
            queue.get_nowait()
        for _ in range(n_step):
            queue.put(0)

        G = 0
        prev_state = env.reset()
        for t in range(1000):
            if render:
                env.render()

            action = get_action(q_value, state_converter, prev_state)
            state, reward, is_terminated, _ = env.step(action)
            reward = -10 if is_terminated and t < 195 else 1 
            G = G * gamma + reward
            queue.put(reward)
            G -= queue.get() * gamma_n

            if t + 1 >= n_step:
                update_q_value(q_value, prev_state, action, state, reward)

            prev_state = state
            if is_terminated:
                print("Episode {} finished after {} timesteps".format(i, t + 1))
                time_steps.append(t + 1)
                break
        
        if (i + 1) % 20 == 0:
            plot(time_steps, 'Time Step', figure_path)

def backward_sarsa(
        env,
        state_converter,
        n_step=1,
        alpha=0.1,
        gamma=0.95,
        episode_count=2000,
        render=False,
        figure_path=None,
    ):
    def update_q_value(q_value, curr_state, curr_action, next_state, next_reward):
        curr_index = state_converter.state_to_index(curr_state)
        next_index = state_converter.state_to_index(next_state)
        q_value[curr_index][curr_action] = \
            (1 - alpha) * q_value[curr_index][curr_action] + \
            alpha * (next_reward + gamma * np.max(q_value[next_index]))

    q_value = state_converter.get_initial_q_value()

    queue = Queue()
    gamma_n = gamma ** n_step

    time_steps = []
    for i in range(episode_count):
        if not queue.empty():
            queue.get_nowait()
        for _ in range(n_step):
            queue.put(0)

        G = 0
        prev_state = env.reset()
        for t in range(1000):
            if render:
                env.render()

            action = get_action(q_value, state_converter, prev_state)
            state, reward, is_terminated, _ = env.step(action)
            reward = -10 if is_terminated and t < 195 else 1 
            G = G * gamma + reward
            queue.put(reward)
            G -= queue.get() * gamma_n

            if t + 1 >= n_step:
                update_q_value(q_value, prev_state, action, state, reward)

            prev_state = state
            if is_terminated:
                print("Episode {} finished after {} timesteps".format(i, t + 1))
                time_steps.append(t + 1)
                break
        
        if (i + 1) % 20 == 0:
            plot(time_steps, 'Time Step', figure_path)

def run():
    set_logger()
    args = get_args()
    set_seed(args.seed)

    figure_path = os.path.join(args.output_dir, args.time_step_plot)
    setup_output_dir(args.output_dir, dict(args._get_kwargs())) #pylint: disable=protected-access

    state_converter = CartPoleStateConverter(epsilon=args.epsilon)

    env = gym.make('CartPole-v0')
    if args.algorithm == 'valueiter':
        value_iteration(
            env,
            state_converter,
            gamma=args.gamma,
            episode_count=args.episode_count,
            render=args.render,
            figure_path=figure_path,
        )
    elif args.algorithm == 'montecarlo':
        monte_carlo(
            env,
            state_converter,
            first_visit=args.first_visit,
            gamma=args.gamma,
            episode_count=args.episode_count,
            render=args.render,
            figure_path=figure_path,
        )
    elif args.algorithm == 'sarsa':
        sarsa(
            env,
            state_converter,
            n_step=args.n_step,
            alpha=args.alpha,
            gamma=args.gamma,
            episode_count=args.episode_count,
            render=args.render,
            figure_path=figure_path,
        )
    elif args.algorithm == 'qlearning':
        q_learning(
            env,
            state_converter,
            n_step=args.n_step,
            alpha=args.alpha,
            gamma=args.gamma,
            episode_count=args.episode_count,
            render=args.render,
            figure_path=figure_path,
        )
    env.close()

if __name__ == '__main__':
    run()
