"""
Calculate a t-SNE plain embedding by visualizing extracted vectors on a plane.
"""
import logging
import dill
import numpy as np
from sklearn.manifold import TSNE

def get_metric(metric):
    """
    Get a metric.
    """
    if metric == 'euclidian':
        def euclidean(v, w):
            return np.sqrt(np.sum((v - w) * (v - w)))
        return euclidean
    if metric == 'cosine':
        def cosine(v, w):
            return np.sum(v * w) / np.sqrt(np.sum(v * v)) / np.sqrt(np.sum(w * w))
        return cosine
    if metric == 'dist_on_poincare_disk':
        def dist_on_poincare_disk(v, w):
            delta = 2 * np.sum((v - w) * (v - w)) / (1 - np.sum(v * v)) / (1 - np.sum(w * w))
            arcosh = lambda x: np.log(x + np.sqrt((x + 1) * (x - 1)))
            return arcosh(1 + delta)
        return dist_on_poincare_disk
    raise ValueError('The metric should be "euclidean", "cosine", or "dist_on_poincare_disk".')

def tsne_embedding(text_vectors, keys, metric, save_plain_embed_path, tsne_params):
    """
    Calculate a t-SNE plain embedding.
    """
    logging.info('Start calculating a t-SNE plain embedding.')

    model = TSNE(**tsne_params, metric=get_metric(metric))
    plain_embeds = model.fit_transform(text_vectors)

    with open(save_plain_embed_path, 'wb') as output_file:
        dill.dump((keys, plain_embeds), output_file)

    logging.info('Finish calculating a plain embedding.')
