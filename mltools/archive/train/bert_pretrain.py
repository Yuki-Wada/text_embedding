"""
Pretrain a BERT model which assign a label to an input text.
"""
import numpy as np
from tqdm import tqdm
import torch

from nlp_model.utils import set_random_seed, get_mb_indices, get_optimizer, use_cuda
from nlp_model.preprocess.text_processor import TextProcessor
from nlp_model.models.bert import BERT
from nlp_model.models.parameter_optimization import HyperParamOptimizer

def run(
        data, text_processor, logger,
        save_model_path, mb_size, epochs, optimizer_params, model_params,
        seed=None):
    """
    Pretrain a BERT model.
    """
    # set NumPy random seed
    set_random_seed(seed)

    first_sentences, second_sentences, labels = data

    data_count = len(labels)
    indices = np.arange(data_count)

    hyper_param_optimizer = HyperParamOptimizer(model_params)
    for optim_number in range(hyper_param_optimizer.optim_count):
        logger.info('Start Optimization %s', optim_number + 1)

        model = BERT(text_processor, use_cuda=use_cuda(), **hyper_param_optimizer.get_next_config())
        optimizer = get_optimizer(model, **optimizer_params)

        best_score = None
        scores = []
        for epoch in range(epochs):
            # Pretrain
            total_loss = 0.0
            total_count = 0
            logger.info('Start Epoch %s', epoch + 1)
            with tqdm(total=data_count, desc='Train') as pbar:
                for mb_indices in get_mb_indices(indices, mb_size, 'shuffle_begin_index'):
                    mb_count = len(mb_indices)
                    mb_first_sentences = first_sentences[mb_indices]
                    mb_second_sentences = second_sentences[mb_indices]
                    mb_labels = labels[mb_indices]

                    try:
                        model.zero_grad()
                        loss = model.pretrain(mb_first_sentences, mb_second_sentences, mb_labels)
                        loss.backward()
                        optimizer.step()

                        mb_average_loss = loss.data.numpy()
                        total_loss += mb_average_loss * mb_count
                        total_count += mb_count

                    except Exception as e:
                        logger.error(str(e))
                        mb_average_loss = np.nan
                    finally:
                        torch.cuda.empty_cache()

                    pbar.update(mb_count)
                    pbar.set_postfix(Loss='{:.4f}'.format(mb_average_loss))

            loss = total_loss / total_count
            logger.info('Train Loss: {:.4f}'.format(loss))

            # Set a score so that a better model will get a higher score.
            curr_score = - loss
            scores.append(curr_score)
            if best_score is None or best_score < curr_score:
                logger.info('The current epoch score is best for the current parameter tuning.')
                best_score = curr_score
                model.save(save_model_path)

            logger.info('End Epoch %s', epoch + 1)

        hyper_param_optimizer.add_score(best_score)

        logger.info('Score: ')
        logger.info(scores)

        logger.info('End Optimization %s', optim_number + 1)

    hyper_param_optimizer.save_optim_result()
