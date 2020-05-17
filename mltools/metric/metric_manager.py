"""
Define a class to manage metrics.
"""
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from mltools.utils import dump_json

logger = logging.getLogger(__name__)

class MetricManager:
    def __init__(self, output_dir_path, epoch_count):
        self.epoch_scores = [{'epoch': epoch + 1} for epoch in range(epoch_count)]
        self.output_dir_path = output_dir_path

    def register_loss(self, loss, epoch, mode):
        if mode not in self.epoch_scores[epoch]:
            self.epoch_scores[epoch][mode] = {}
        self.epoch_scores[epoch][mode]['loss'] = loss

    def register_confusion_matrix(self, confusion_matrix, epoch, mode):
        if mode not in self.epoch_scores[epoch]:
            self.epoch_scores[epoch][mode] = {}
        self.epoch_scores[epoch][mode]['confusion_matrix'] = confusion_matrix.tolist()

    def save_score(self):
        dump_json(
            self.epoch_scores,
            os.path.join(
                self.output_dir_path,
                'score.json',
            ),
        )

    def plot_loss(self, label, figure_path):
        metric_dict = {}
        for epoch_dict in self.epoch_scores:
            for mode in epoch_dict:
                if mode in ['epoch']:
                    continue
                if mode not in metric_dict:
                    metric_dict[mode] = []
                metric_dict[mode].append(epoch_dict[mode]['loss'])

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        for mode, metric in metric_dict.items():
            ax.plot(np.arange(len(metric)) + 1, metric, label=mode)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(label)
        ax.legend()

        x_ax = ax.get_xaxis()
        x_ax.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(figure_path)
        plt.close()
