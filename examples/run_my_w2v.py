"""
Run my Word2Vec model.
"""
import logging
import argparse
import dill
from gensim.corpora import Dictionary
from tqdm import tqdm

from mltools.utils import set_seed, set_logger
from mltools.dataset.wikipedia_ja import WikipediaDataSet, WikipediaDataLoader
from mltools.model.word2vec import MyWord2Vec

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input_dir", type=str, help="input directory path", required=True)
parser.add_argument(
    "--cache_dir", type=str, help="directory to cache data set", required=True)
parser.add_argument("--model_name_to_save", type=str, help="model path to save")

parser.add_argument("--window", type=int, default=5, help="The window size of skip-gram")
parser.add_argument("--size", type=int, default=100, help="The dimension of word representation")
parser.add_argument(
    "--negative", type=int, default=5, help="The number per word of negative samples to use")
parser.add_argument(
    "--ns_exponent", type=float, default=0.75,
    help="The exponent used to shape the negative sampling distribution.")

parser.add_argument(
    "--min_count", type=int, default=5,
    help="Ignores all words with total frequency lower than this")

parser.add_argument(
    "--alpha", type=float, default=0.025, help="learning rate in the final epoch")
parser.add_argument(
    "--min_alpha", type=float, default=0.0001, help="learning rate in the final epoch")

parser.add_argument("--epochs", type=int, default=20, help="epoch count")
parser.add_argument("--mb_size", type=int, default=512, help="minibatch size")
parser.add_argument("--workers", type=int, default=1, help="the number of core to use")
parser.add_argument("--seed", type=int, help="random seed for initialization")

args = parser.parse_args()

logger = logging.getLogger(__name__)

def run():
    set_seed(args.seed)
    set_logger()

    logger.info('Load Wikipedia articles in Japanese.')

    data_set = WikipediaDataSet(args.input_dir, args.cache_dir)
    data_loader = WikipediaDataLoader(data_set, args.mb_size * args.workers)

    dictionary: Dictionary = data_set.dictionary
    dictionary.filter_extremes(no_below=args.min_count, no_above=0.999)
    w2v_model = MyWord2Vec(
        dictionary=dictionary,
        window=args.window,
        size=args.size,
        negative=args.negative,
        ns_exponent=args.ns_exponent,
        alpha=args.alpha,
        workers=args.workers)

    logger.info('Train my Word2Vec model.')
    for epoch in range(args.epochs):
        logger.info('Epoch: %d', epoch + 1)

        w2v_model.wc = 0
        with tqdm(total=len(data_set), desc="Train Word2Vec") as pbar:
            for mb_texts in data_loader.get_iter():
                mb_indexed_texts = [dictionary.doc2idx(text) for text in mb_texts]
                w2v_model.train(mb_indexed_texts)

                w2v_model.lr = \
                    args.alpha - ((args.alpha - args.min_alpha) * (epoch + 1) / args.epochs)
                pbar.update(len(mb_indexed_texts))

        if args.model_name_to_save:
            logger.info('Save my Word2Vec model.')
            with open(args.model_name_to_save, 'wb') as _:
                dill.dump(w2v_model, _)

if __name__ == '__main__':
    run()
