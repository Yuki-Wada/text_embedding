"""
Run gensim Word2Vec model.
"""
import logging
import argparse
import dill
from gensim.models.word2vec import Word2Vec

from mltools.utils import set_seed, set_logger
from mltools.dataset.wikipedia_ja import WikipediaDataSet

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
parser.add_argument("--workers", type=int, default=4, help="the number of core to use")
parser.add_argument("--seed", type=int, help="random seed for initialization")

args = parser.parse_args()

logger = logging.getLogger(__name__)

def run():
    set_seed(args.seed)
    set_logger()

    logger.info('Load Wikipedia articles in Japanese.')

    data_set = WikipediaDataSet(args.input_dir, args.cache_dir)

    logger.info('Train gensim Word2Vec model.')
    w2v_model = Word2Vec(
        window=args.window,
        size=args.size,
        negative=args.negative,
        ns_exponent=args.ns_exponent,
        min_count=args.min_count,
        alpha=args.alpha,
        min_alpha=args.min_alpha,
        iter=args.epochs,
        workers=args.workers,
        seed=args.seed)
    w2v_model.build_vocab(list(data_set.get_text()))
    w2v_model.train(list(data_set.get_text()), total_examples=len(data_set), epochs=args.epochs)

    if args.model_name_to_save:
        logger.info('Save gensim Word2Vec model.')
        with open(args.model_name_to_save, 'wb') as _:
            dill.dump(w2v_model, _)

if __name__ == '__main__':
    run()
