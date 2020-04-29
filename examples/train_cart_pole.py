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

    parser.add_argument("--episode_count", type=int, default=10000000)
    parser.add_argument("--plot_interval", type=int, default=10000)
    parser.add_argument("--plot_count", type=int, default=10000)

    parser.add_argument("--alpha", type=float, default=1e-1)
    parser.add_argument("--discount", type=float, default=0.99)

    parser.add_argument("--render", action='store_true')

    parser.add_argument("--seed", type=int)

    args = parser.parse_args()

    return args

def plot(values, plot_count, label, figure_path):
    episode_numbers = np.arange(0, len(values), int(len(values) / plot_count))

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

class CartPoleQValue:
    def __init__(self, action_num=2, alpha=1e-1, discount=0.99):
        self.action_num = action_num
        self.alpha = alpha
        self.discount = discount

        self.min_position = -2.4
        self.max_position = 2.4
        self.position_unit = 0.1
        self.position_size = int(
            (self.max_position - self.min_position) / self.position_unit)

        self.min_velocity = -3.0
        self.max_velocity = 3.0
        self.velocity_unit = 0.1
        self.velocity_size = int(
            (self.max_velocity - self.min_velocity) / self.velocity_unit)

        self.min_angle = -2.4
        self.max_angle = 2.4
        self.angle_unit = 0.1
        self.angle_size = int(
            (self.max_angle - self.min_angle) / self.angle_unit)

        self.min_angular_velocity = -2.0
        self.max_angular_velocity = 2.0
        self.angular_velocity_unit = 0.1
        self.angular_velocity_size = int(
            (self.max_angular_velocity - self.min_angular_velocity) / self.angular_velocity_unit)

        self.q_value = np.random.random((
            self.position_size,
            self.velocity_size,
            self.angle_size,
            self.angular_velocity_size,
            self.action_num,
        )) + 1

    def state_to_index(self, state, action=None):
        position, velocity, angle, angular_velocity = state
        position_index = int((position - self.min_position) / self.position_unit)
        velocity_index = int((velocity - self.min_velocity) / self.velocity_unit)
        angle_index = int((angle - self.min_angle) / self.angle_unit)
        angular_velocity_index = int(
            (angular_velocity - self.min_angular_velocity) / self.angular_velocity_unit)

        position_index = min(position_index, self.position_size - 1)
        velocity_index = min(velocity_index, self.velocity_size - 1)
        angle_index = min(angle_index, self.angle_size - 1)
        angular_velocity_index = min(angular_velocity_index, self.angular_velocity_size - 1)

        if action:
            return position_index, velocity_index, angle_index, angular_velocity_index, action
        return position_index, velocity_index, angle_index, angular_velocity_index

    def get_q_value(self, state, action=None):
        index = self.state_to_index(state, action)
        return self.q_value[index]

    def get_action(self, state):
        return np.argmax(self.get_q_value(state))

    def update_q_value(self, curr_state, curr_action, next_state, next_reward):
        curr_index = self.state_to_index(curr_state, curr_action)
        self.q_value[curr_index] = (1 - self.alpha) * self.q_value[curr_index] + \
            self.alpha * (next_reward + self.discount * np.max(self.get_q_value(next_state)))

        return self.q_value[curr_index]

def setup_output_dir(output_dir_path, args):
    os.makedirs(output_dir_path, exist_ok=True)
    dump_json(args, os.path.join(output_dir_path, 'args.json'))

def run():
    set_logger()
    args = get_args()
    set_seed(args.seed)

    figure_path = os.path.join(args.output_dir, args.time_step_plot)
    setup_output_dir(args.output_dir, dict(args._get_kwargs())) #pylint: disable=protected-access

    q_value = CartPoleQValue(alpha=args.alpha, discount=args.discount)

    env = gym.make('CartPole-v0')
    values = []
    for i in range(args.episode_count):
        prev_state = env.reset()
        for t in range(1000):
            if args.render:
                env.render()

            action = q_value.get_action(prev_state)
            state, reward, is_terminated, _ = env.step(action)

            q_value.update_q_value(prev_state, action, state, reward)

            prev_state = state
            if is_terminated:
                print("Episode {} finished after {} timesteps".format(i, t + 1))
                values.append(t + 1)
                break

        if (i + 1) % args.plot_interval == 0:
            plot(values, args.plot_count, 'Time Step', figure_path)

    env.close()

if __name__ == '__main__':
    run()
