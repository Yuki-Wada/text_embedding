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
        self.ja_dictionary = Dictionary([['<PAD>'], ['<BOS>'], ['<UNK>']])
        self.en_dictionary = Dictionary([['<PAD>'], ['<BeginOfEncode>'], ['<BOS>'], ['<EOS>'], ['<UNK>']])
        self.is_training = is_training

    def register_ja_texts(self, texts: List[List[str]]):
        if self.is_training:
            self.ja_dictionary.add_documents(texts)

    def register_en_texts(self, texts: List[List[str]]):
        if self.is_training:
            self.en_dictionary.add_documents(texts)

    @property
    def ja_unknown_word_index(self):
        return self.ja_dictionary.token2id['<UNK>']

    @property
    def en_unknown_word_index(self):
        return self.en_dictionary.token2id['<UNK>']

    @property
    def ja_vocab_count(self):
        return len(self.ja_dictionary)

    @property
    def en_vocab_count(self):
        return len(self.en_dictionary)

    def doc2idx_ja(self, texts):
        return self.ja_dictionary.doc2idx(texts, unknown_word_index=self.ja_unknown_word_index)

    def doc2idx_en(self, texts):
        return self.en_dictionary.doc2idx(texts, unknown_word_index=self.en_unknown_word_index)

class BilingualDataSet:
    def __init__(
            self,
            sort_by_length: bool = True,
            is_training: bool = False,
            preprocessor: BilingualPreprocessor = None
        ):

        self.sort_by_length = sort_by_length

        if preprocessor is None:
            self.preprocessor = BilingualPreprocessor(is_training=is_training)
        else:
            self.preprocessor = preprocessor
            self.preprocessor.is_training = is_training

        self.data = []

    def input_data(self, file_paths):
        for file_path in file_paths:
            with open(file_path, 'rb') as _:
                data = dill.load(_)
            data = [
                datum for datum in data
                if len(datum['japanese']) > 5 and len(datum['english']) > 5]
            ja_texts = [pair['japanese'] for pair in data]
            en_texts = [pair['english'] for pair in data]
            self.preprocessor.register_ja_texts(ja_texts)
            self.preprocessor.register_en_texts(en_texts)
            self.data += data
            if self.sort_by_length:
                self.data = sorted(self.data, key=lambda x: len(x['english']))

    def __getitem__(self, index):
        return self.data[index]['japanese'], self.data[index]['english']

    def __iter__(self):
        if self.sort_by_length:
            data = sorted(data, key=lambda x: len(x[0]) + len(x[1]))
        for datum in np.random.permutation(data):
            yield datum['japanese'], datum['english']

    @property
    def ja_vocab_count(self):
        return self.preprocessor.ja_vocab_count

    @property
    def en_vocab_count(self):
        return self.preprocessor.en_vocab_count

    def __len__(self) -> int:
        return len(self.data)

    def doc2idx_ja(self, texts):
        return self.preprocessor.doc2idx_ja(texts)

    def doc2idx_en(self, texts):
        return self.preprocessor.doc2idx_en(texts)

class BilingualDataLoader:
    def __init__(self, data_set: BilingualDataSet, mb_size: int, do_shuffle: bool = True):
        self.data_set = data_set
        self.mb_size = mb_size
        self.do_shuffle = do_shuffle

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

        indices = np.arange(len(self.data_set))
        begin_indices = np.arange(0, len(self.data_set), self.mb_size)
        if self.do_shuffle:
            begin_indices = np.random.permutation(begin_indices)

        for begin_index in begin_indices:
            ja_texts = []
            en_texts = []
            for index in indices[begin_index : begin_index + self.mb_size]:
                ja_text, en_text = self.data_set[index]
                en_text = ['<BeginOfEncode>'] + en_text + ['<EOS>']
                ja_texts.append(self.doc2idx_ja(ja_text))
                en_texts.append(self.doc2idx_en(en_text))

            ja_text_array = self.texts_to_array(ja_texts)
            en_text_array = self.texts_to_array(en_texts)

            yield ja_text_array, en_text_array

        ja_text_array = self.texts_to_array(ja_texts)
        en_text_array = self.texts_to_array(en_texts)

        yield ja_text_array, en_text_array
