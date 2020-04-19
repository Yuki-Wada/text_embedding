"""
Preprocess Japanese-English Bilingual Corpus.
"""
import os
import argparse
import logging
import glob
import xml.etree.ElementTree as ET
import dill
import nltk
import MeCab

logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input_dir',
        default='data/original/japaneseenglish-bilingual-corpus/wiki_corpus_2.01'
    )
    parser.add_argument('--cache_dir', default='data/cache/japaneseenglish-bilingual-corpus')
    parser.add_argument('--ratios', nargs='+', type=float, default=[0.8, 0.2])
    parser.add_argument('--file_names', nargs='+', default=['train', 'ratio'])

    args = parser.parse_args()

    return args

def tokenize_ja(text, tokenizer: MeCab.Tagger):
    words = []
    word_infos = tokenizer.parse(text).split('\n')[:-2]
    for word_info in word_infos:
        word_info = word_info.split('\t')
        words.append(word_info[2])
    return words

def tokenize_en(text):
    return nltk.word_tokenize(text)

def category_to_data(category_dir_path: str, ja_tokenizer: MeCab.Tagger):
    data = []
    for xml_path in glob.glob('{}/*.xml'.format(category_dir_path)):
        try:
            tree = ET.parse(xml_path)
            for sentence_element in tree.getiterator('sen'):
                ja_en_pair = {}
                for j in sentence_element.iter('j'):
                    ja_en_pair['japanese'] = tokenize_ja(j.text, ja_tokenizer)
                for e in sentence_element.iter('e'):
                    if e.attrib['type'] == 'check':
                        ja_en_pair['english'] = tokenize_en(e.text)
                data.append(ja_en_pair)
        except Exception as e: #pylint: disable=broad-except
            logger.error(str(e))

    return data

def preprocess_bilingual_data_set(input_dir_path: str, cache_dir_path: str, ratios, file_names):
    ja_tokenizer = MeCab.Tagger('-Ochasen')
    os.makedirs(cache_dir_path, exist_ok=True)

    data = []
    for category_dir_path in glob.glob('{}/*'.format(input_dir_path))[1:]:
        rel_path = os.path.relpath(category_dir_path, input_dir_path)
        dst_dir_path = os.path.join(cache_dir_path, rel_path)
        if os.path.isdir(dst_dir_path):
            continue

        os.makedirs(dst_dir_path, exist_ok=True)
        data = category_to_data(category_dir_path, ja_tokenizer)

        cum_ratio = 0.0
        for ratio, file_name in zip(ratios, file_names):
            dst_file_path = os.path.join(dst_dir_path, file_name)
            with open(dst_file_path, 'wb') as _:
                dill.dump(data[int(len(data) * cum_ratio): int(len(data) * (cum_ratio + ratio))], _)
            cum_ratio += ratio

def run():
    args = get_args()

    assert len(args.ratios) == len(args.file_names), \
        'The length of ratios should be equal to that of file_names.'

    preprocess_bilingual_data_set(
        args.input_dir_path,
        args.cache_dir_path,
        args.ratios,
        args.file_names
    )

if __name__ == '__main__':
    run()
