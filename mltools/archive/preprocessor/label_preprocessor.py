"""
Define a class which converts and tokenizes a label.
"""
from copy import deepcopy
import dill

class LabelConverter:
    """
    Define a class which converts and tokenizes a label.
    """
    def __init__(self, label_dict, unknown_label='その他'):
        self.label_dict = deepcopy(label_dict)
        self.unknown_label = unknown_label

        self.token_to_converted_label = {}
        self.converted_label_to_token = {}
        for i, converted_label in enumerate(set(label_dict.values())):
            self.token_to_converted_label[i] = converted_label
            self.converted_label_to_token[converted_label] = i

        if self.unknown_label not in self.converted_label_to_token:
            unknown_token = len(self.token_to_converted_label)
            self.token_to_converted_label[unknown_token] = self.unknown_label
            self.converted_label_to_token[self.unknown_label] = unknown_token

    @property
    def label_count(self):
        return len(self.converted_label_to_token)

    def convert(self, label):
        """
        Convert a label.
        """
        if label in self.label_dict:
            converted_label = self.label_dict[label]
        else:
            converted_label = self.unknown_label

        return converted_label

    def tokenize(self, converted_label):
        """
        Tokenize a converted label into a token.
        """
        return self.converted_label_to_token[converted_label]

    def detokenize(self, token):
        """
        Detokenize a token into a converted label.
        """
        return self.token_to_converted_label[token]

    def before_serialize(self):
        """
        You must execute this function before serializing this model.
        """

    def after_deserialize(self):
        """
        You must execute this function after deserializing into this model.
        """

    def save(self, save_path):
        """
        Save the model by serializing it to a pickled file.
        """
        self.before_serialize()
        with open(save_path, 'wb') as _:
            dill.dump(self, _)

    @staticmethod
    def load(load_path):
        """
        Load the model by deserializing a loaded pickled file.
        """
        with open(load_path, 'rb') as _:
            label_converter = dill.load(_)
        assert isinstance(label_converter, LabelConverter), 'LabelConverter モデルのファイルパスを指定してください。'
        label_converter.after_deserialize()
        return label_converter
