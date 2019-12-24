import os
import bz2
import logging
import re
import dill
import numpy as np
from gensim.corpora import Dictionary

import MeCab

logger = logging.getLogger(__name__)

class WikipediaDataSet:
    def __init__(self, src_dir_path: str, cache_dir_path: str):
        self.src_dir_path = src_dir_path
        self.cache_dir_path = cache_dir_path
        self.dictionary = Dictionary()
        self.cache_file_paths = []

        tokenizer = MeCab.Tagger('-Ochasen')
        self.load_file(tokenizer)

    @staticmethod
    def tokenize(tokenizer: MeCab.Tagger, text):
        words = []
        word_infos = tokenizer.parse(text).split('\n')[:-2]
        for word_info in word_infos:
            word_info = word_info.split('\t')
            if '名詞' in word_info[3] or '動詞' in word_info[3] or '形容詞' in word_info[3]:
                words.append(word_info[2])
        return words

    @staticmethod
    def article_to_words(tokenizer: MeCab.Tagger, article: str):
        match = re.search(r'\<doc(.|\s)*?\>\n', article)
        article = article[match.end():]
        match = re.search(r'\</doc>', article)
        article = article[:match.start()]

        texts = []
        for line in article.split('\n'):
            if not line:
                continue
            texts.append(WikipediaDataSet.tokenize(tokenizer, line))

        return texts

    def load_file(self, tokenizer: MeCab.Tagger):
        os.makedirs(self.cache_dir_path, exist_ok=True)

        for subdir_name in os.listdir(self.src_dir_path):
            subdir_path = os.path.join(self.src_dir_path, subdir_name)
            file_path_to_save = os.path.join(self.cache_dir_path, subdir_name)
            if os.path.exists(file_path_to_save):
                with open(file_path_to_save, 'rb') as _:
                    texts = dill.load(_)
            else:
                texts = []
                for file_name in os.listdir(subdir_path):
                    file_path = os.path.join(subdir_path, file_name)
                    with bz2.open(file_path, 'r') as _:
                        raw_articles = _.read().decode('utf-8')

                    match = re.search(r'\<doc(.|\s)*?\</doc>\n', raw_articles)
                    while match:
                        start, end = match.span()
                        article = raw_articles[start: end]
                        texts += WikipediaDataSet.article_to_words(tokenizer, article)
                        raw_articles = raw_articles[end:]
                        match = re.search(r'\<doc(.|\s)*?\</doc>\n', raw_articles)

                file_path_to_save = os.path.join(self.cache_dir_path, subdir_name)
                with open(file_path_to_save, 'wb') as _:
                    dill.dump(texts, _)

            self.dictionary.add_documents(texts)
            self.cache_file_paths.append(file_path_to_save)

    def get_text(self):
        for file_path_to_load in np.random.permutation(self.cache_file_paths):
            with open(file_path_to_load, 'rb') as _:
                texts = dill.load(_)
            for text in np.random.permutation(texts):
                yield text

    def __len__(self) -> int:
        return self.dictionary.num_docs

class WikipediaDataLoader:
    def __init__(self, data_set: WikipediaDataSet, mb_size: int):
        self.data_set = data_set
        self.mb_size = mb_size

    def get_iter(self):
        texts = []
        for text in self.data_set.get_text():
            texts.append(text)
            if len(texts) >= self.mb_size:
                yield texts
                texts = []
        yield texts
