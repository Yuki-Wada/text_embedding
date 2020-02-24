"""
Define a random forest wrapper whose core random forest model is in scikit-learn.
"""
import dill
import numpy as np

from sklearn.ensemble import RandomForestClassifier

class RandomForestWrapper:
    """
    Define a random forest wrapper class.
    """
    def __init__(
            self, text_processor, label_count,
            max_depth, n_estimators, random_state):
        self.random_forest = RandomForestClassifier(
            max_depth=max_depth,
            n_estimators=n_estimators,
            random_state=random_state)
        self.text_processor = text_processor
        self.label_count = label_count

    @property
    def vocab_count(self):
        """
        Return the count of vocabularies used.
        """
        return self.text_processor.vocab_count

    def fit(self, freq_table, labels):
        """
        Train the model by using pairs of a text and a label.
        """
        self.random_forest.fit(freq_table, labels)

    def predict(self, freq_table):
        """
        Assign a label to an input text.
        """
        proba = np.zeros((freq_table.shape[0], self.label_count))
        proba[:, self.random_forest.classes_] = self.random_forest.predict_proba(freq_table)
        return proba

    def before_serialize(self):
        """
        You must execute this function before serializing this model.
        """
        self.text_processor.before_serialize()

    def after_deserialize(self):
        """
        You must execute this function after deserializing into this model.
        """
        self.text_processor.after_deserialize()

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
            model = dill.load(_)
        assert isinstance(model, RandomForestWrapper), 'RandomForestWrapper モデルのファイルパスを指定してください。'
        model.after_deserialize()
        return model
