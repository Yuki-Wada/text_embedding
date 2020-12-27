"""
Train in the Maze Task.
"""

import os
import argparse
import logging
from itertools import product
import random
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
    parser.add_argument('--maze', default='examples/maze.txt')

    parser.add_argument("--iter_count", type=int, default=100)

    parser.add_argument("--algorithm", default='valueiter')
    parser.add_argument("--n_step", type=int, default=1)
    parser.add_argument("--first_visit", action='store_true')

    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.1)

    parser.add_argument("--render", action='store_true')

    parser.add_argument("--seed", type=int)

    args = parser.parse_args()

    return args

def setup_output_dir(output_dir_path, args):
    os.makedirs(output_dir_path, exist_ok=True)
    dump_json(args, os.path.join(output_dir_path, 'args.json'))

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

class MazeEnvironment: 
    def __init__(self, maze_file):
        self.row = 0
        self.col = 0
        self.lines = []
        with open(maze_file, 'r') as f:
            for line in f.readlines():
                line = line[:-1]
                self.lines.append(line)
                self.col = max(self.col, len(line))
                self.row += 1
        
        self.maze = np.zeros((self.row, self.col), dtype=np.uint8)
        self.start = None
        self.goal = None
        for i, line in enumerate(self.lines):
            for j, ch in enumerate(line):
                if ch == '#':
                    self.maze[i, j] = 1
                elif ch == 'S':
                    if self.start is not None:
                        raise ValueError('It is not permitted that multiple starts exist.')
                    self.start = (i, j)
                elif ch == 'G':
                    if self.goal is not None:
                        raise ValueError('It is not permitted that multiple goals exist.')
                    self.goal = (i, j)
        
        if self.start is None:
            raise ValueError('It is not permitted that no start exists.')
        if self.goal is None:
            raise ValueError('It is not permitted that no goal exists.')
        
        self.state = self.start

    def reset(self):
        self.state = self.start
        return self.state

    def locate(self, state):
        if self.maze[tuple(state)] == 1:
            raise ValueError('The agent cannot live in a wall.')
        self.state = state

    def step(self, direction):
        reward = 0
        info = {'effective_action': False}
        if direction == 'R':
            if self.state[1] + 1 < self.col:
                if self.maze[self.state[0], self.state[1] + 1] == 0:
                    self.state = (self.state[0], self.state[1] + 1)
                    reward = -1
                    info['effective_action'] = True
        elif direction == 'L':
            if self.state[1] > 0:
                if self.maze[self.state[0], self.state[1] - 1] == 0:
                    self.state = (self.state[0], self.state[1] - 1)
                    reward = -1
                    info['effective_action'] = True
        elif direction == 'D':
            if self.state[0] + 1 < self.row:
                if self.maze[self.state[0] + 1, self.state[1]] == 0:
                    self.state = (self.state[0] + 1, self.state[1])
                    reward = -1
                    info['effective_action'] = True
        elif direction == 'U':
            if self.state[0] > 0:
                if self.maze[self.state[0] - 1, self.state[1]] == 0:
                    self.state = (self.state[0] - 1, self.state[1])
                    reward = -1
                    info['effective_action'] = True
        else:
            raise ValueError('The args "direction" should be "R", "L", "D" or "U" either.')

        is_terminated = self.state == self.goal

        return (self.state, reward, is_terminated, info)

    def render(self):
        for line in self.lines:
            print(line)

    def close(self):
        pass

    def is_goal(self, state):
        return self.goal == state

    def is_wall(self, state):
        return self.maze[state] == 1

    def get_initial_q_value(self):
        q_value_shape = tuple(list(self.maze.shape + [4]))
        q_value = np.random.uniform(low=0, high=1, size=q_value_shape)
        return q_value

    def get_initial_v_value(self):
        v_value = np.random.uniform(low=0, high=1, size=self.maze.shape)
        v_value[self.maze == 1] = -1000
        v_value[self.goal] = 0
        return v_value

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

def get_action_for_v_value(env, v_value, state):
    directions = ['R', 'L', 'D', 'U']
    max_v = None
    for direction in ['R', 'L', 'D', 'U']:
        next_state, _, _, info = env.step(direction)
        if info['effective_action'] and (max_v is None or max_v < v_value[next_state]):
            action = direction
            max_v = v_value[next_state]
        env.locate(state)

    return action

def value_iteration(
        env: MazeEnvironment,
        gamma=0.95,
        iter_count=2000,
        render=False,
        figure_path=None,
    ):
    def update_v_value(env, v_value):
        for i in range(env.row):
            for j in range(env.col):
                state = (i, j)
                if env.is_wall(state) or env.is_goal(state):
                    continue
                new_vs = []
                for direction in ['R', 'L', 'D', 'U']:
                    env.locate(state)
                    next_state, reward, _, info = env.step(direction)
                    if info['effective_action']:
                        new_vs.append(reward + gamma * v_value[next_state])
                v_value[state] = max(new_vs)

    v_value = env.get_initial_v_value()

    max_time_step = 1000
    time_steps = []
    for i in range(iter_count):
        update_v_value(env, v_value)

        prev_state = env.reset()
        for t in range(max_time_step):
            if render:
                pass
                # env.render()

            action = get_action_for_v_value(env, v_value, prev_state)
            state, _, is_terminated, _ = env.step(action)

            prev_state = state
            if is_terminated:
                print(f'Iteration {i} finished after {t + 1} timesteps')
                time_steps.append(t + 1)
                break
        else:
            print(f'Iteration {i} not finished after {max_time_step} timesteps')

        if (i + 1) % 20 == 0:
            plot(time_steps, 'Time Step', figure_path)

def run():
    set_logger()
    args = get_args()
    set_seed(args.seed)

    figure_path = os.path.join(args.output_dir, args.time_step_plot)
    setup_output_dir(args.output_dir, dict(args._get_kwargs())) #pylint: disable=protected-access

    env = MazeEnvironment(args.maze)
    if args.algorithm == 'valueiter':
        value_iteration(
            env,
            gamma=args.gamma,
            iter_count=args.iter_count,
            render=args.render,
            figure_path=figure_path,
        )
    env.close()

if __name__ == '__main__':
    run()
