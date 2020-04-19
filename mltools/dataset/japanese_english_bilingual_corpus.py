"""
Define a Data Set Class to Preprocess Japanese-English Bilingual Corpus.
"""
from typing import List
import logging
import dill
import numpy as np
from gensim.corpora import Dictionary

logger = logging.getLogger(__name__)

class BilingualPreprocessor:
    def __init__(self, is_training=False):
        self.ja_dictionary = Dictionary([['<PAD>', '<BOS>']])
        self.en_dictionary = Dictionary([['<PAD>', '<BeginOfEncode>', '<BOS>', '<EOS>']])
        self.is_training = is_training

    def register_ja_texts(self, texts: List[List[str]]):
        if self.is_training:
            self.ja_dictionary.add_documents(texts)

    def register_en_texts(self, texts: List[List[str]]):
        if self.is_training:
            self.en_dictionary.add_documents(texts)

    @property
    def ja_vocab_count(self):
        return len(self.ja_dictionary)

    @property
    def en_vocab_count(self):
        return len(self.en_dictionary)

    def __len__(self) -> int:
        return self.ja_dictionary.num_docs

    def doc2idx_ja(self, texts):
        return self.ja_dictionary.doc2idx(texts)

    def doc2idx_en(self, texts):
        return self.en_dictionary.doc2idx(texts)

class BilingualDataSet:
    def __init__(
            self,
            sort_by_length: bool = True,
            is_training: bool = False,
            preprocessor: BilingualPreprocessor = None
        ):

        self.cache_file_paths = []
        self.sort_by_length = sort_by_length

        if preprocessor is None:
            self.preprocessor = BilingualPreprocessor(is_training=is_training)
        else:
            self.preprocessor = preprocessor
            self.preprocessor.is_training = is_training

    def input_data(self, file_paths):
        for file_path in file_paths:
            with open(file_path, 'rb') as _:
                data = dill.load(_)
            ja_texts = [pair['japanese'] for pair in data]
            en_texts = [pair['english'] for pair in data]
            self.preprocessor.register_ja_texts(ja_texts)
            self.preprocessor.register_en_texts(en_texts)
            self.cache_file_paths.append(file_path)

    def __iter__(self):
        for file_path_to_load in np.random.permutation(self.cache_file_paths):
            with open(file_path_to_load, 'rb') as _:
                data = dill.load(_)
            for datum in np.random.permutation(data):
                yield datum['japanese'], datum['english']

    @property
    def ja_vocab_count(self):
        return self.preprocessor.ja_vocab_count

    @property
    def en_vocab_count(self):
        return self.preprocessor.en_vocab_count

    def __len__(self) -> int:
        return len(self.preprocessor)

    def doc2idx_ja(self, texts):
        return self.preprocessor.doc2idx_ja(texts)

    def doc2idx_en(self, texts):
        return self.preprocessor.doc2idx_en(texts)

class BilingualDataLoader:
    def __init__(self, data_set: BilingualDataSet, mb_size: int):
        self.data_set = data_set
        self.mb_size = mb_size

    @staticmethod
    def texts_to_array(texts):
        count = len(texts)
        max_length = np.max([len(text) for text in texts])
        text_array = np.zeros((count, max_length), dtype=np.int32)
        for i, text in enumerate(texts):
            for j, word in enumerate(text):
                text_array[i, j] = word
        return text_array

    @property
    def ja_vocab_count(self):
        return self.data_set.ja_vocab_count

    @property
    def en_vocab_count(self):
        return self.data_set.en_vocab_count

    def __len__(self) -> int:
        return len(self.data_set)

    def doc2idx_ja(self, texts):
        return self.data_set.doc2idx_ja(texts)

    def doc2idx_en(self, texts):
        return self.data_set.doc2idx_en(texts)

    def __iter__(self):
        ja_texts = []
        en_texts = []
        count = 0
        for ja_text, en_text in self.data_set:
            ja_texts.append(self.doc2idx_ja(ja_text))
            en_texts.append(self.doc2idx_en(en_text))
            count += 1
            if count >= self.mb_size:
                ja_text_array = self.texts_to_array(ja_texts)
                en_text_array = self.texts_to_array(en_texts)

                yield ja_text_array, en_text_array

                ja_texts = []
                en_texts = []
                count = 0

        ja_text_array = self.texts_to_array(ja_texts)
        en_text_array = self.texts_to_array(en_texts)

        yield ja_text_array, en_text_array
