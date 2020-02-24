"""
Pretrain an ELMo model.
"""
import numpy as np
from tqdm import tqdm
import torch

from nlp_model.utils import set_random_seed, get_mb_indices, get_optimizer, use_cuda
from nlp_model.preprocess.text_processor import TextProcessor
from nlp_model.models.elmo import ELMo

def run(
        texts, text_processor, logger,
        save_model_path, epochs, mb_size, optimizer_params, model_params,
        seed=None):
    """
    Pretrain an ELMo model.
    """
    # set NumPy random seed
    set_random_seed(seed)

    train_data_count = len(texts)
    train_indices = np.arange(train_data_count)

    model = ELMo(text_processor, use_cuda=use_cuda(), **model_params)
    optimizer = get_optimizer(model, **optimizer_params)

    best_score = None
    for epoch in range(epochs):
        total_train_loss = 0.0
        total_count = 0
        logger.info('Start Epoch %s', epoch + 1)
        with tqdm(total=train_data_count, desc='Train') as pbar:
            for indices in get_mb_indices(train_indices, mb_size):
                mb_count = len(indices)
                mb_texts = np.array(texts)[indices]

                try:
                    model.zero_grad()
                    loss = model.fit(mb_texts)
                    loss.backward()
                    optimizer.step()

                    mb_average_loss = loss.data.numpy()
                    total_train_loss += mb_average_loss * mb_count
                    total_count += mb_size

                except Exception as e:
                    logger.error(str(e))
                    mb_average_loss = np.nan
                finally:
                    torch.cuda.empty_cache()

                pbar.update(mb_count)
                pbar.set_postfix(Loss='{:.4f}'.format(mb_average_loss))

        train_loss = total_train_loss / total_count
        logger.info('Train Loss: {:.4f}'.format(train_loss))

        # Set a score so that a better model will get a higher score.
        current_score = - train_loss
        if best_score is None or best_score < current_score:
            logger.info('The current epoch score is best for the current parameter tuning.')
            best_score = current_score
            model.save(save_model_path)

        logger.info('End Epoch %s', epoch + 1)
