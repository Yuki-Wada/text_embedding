"""
Train a Poincare embedding model.
"""
from nlp_model.utils import Logger
from nlp_model.preprocess.text_processor import TextProcessor
from nlp_model.models.poincare_embedding import PoincareEmbedding

def run(
        texts, save_model_path, processor_params, initialize_params, train_params):
    """
    Train a Poincare embedding model.
    """
    logger = Logger('Poincare Embedding Train')

    logger.info('Start training a Poincare embedding model.')

    text_processor = TextProcessor(**processor_params)
    texts = texts.apply(text_processor.tokenize)
    texts = texts.apply(text_processor.add_bos_and_eos)
    text_processor.add_documents(texts)

    model = PoincareEmbedding(text_processor, texts, **initialize_params)
    model.train(train_params)

    model.save(save_model_path)

    logger.info('Finish training.')
