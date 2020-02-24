import os
import random

class IMDBProcessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir

        self.data = []

        pos_dir = os.path.join(data_dir, 'pos')
        for pos_file_name in os.listdir(pos_dir):
            pos_file_path = os.path.join(pos_dir, pos_file_name)
            with open(pos_file_path, 'r', encoding='utf8') as _:
                sentence = _.read()
                self.data.append((sentence, 'pos'))

        neg_dir = os.path.join(data_dir, 'neg')
        for neg_file_name in os.listdir(neg_dir):
            neg_file_path = os.path.join(neg_dir, neg_file_name)
            with open(neg_file_path, 'r', encoding='utf8') as _:
                sentence = _.read()
                self.data.append((sentence, 'neg'))

        self.data = random.sample(self.data, len(self.data))

    def get_example(self, index):
        return self.data[index]
