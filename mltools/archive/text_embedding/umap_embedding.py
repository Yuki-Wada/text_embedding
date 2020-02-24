"""
Calculate a UMAP plain embedding by visualizing extracted vectors on a plane.
"""
import logging
import dill
import numpy as np

import umap
import numba

def get_umap_model(metric):
    """
    Get UMAP model.
    """
    if metric == 'euclidian':
        return umap.UMAP(metric='euclidean')
    if metric == 'cosine':
        return umap.UMAP(metric='cosine')
    if metric == 'dist_on_poincare_disk':
        @numba.njit(fastmath=True)
        def dist_on_poincare_disk(v, w):
            delta = 2 * np.sum((v - w) * (v - w)) / (1 - np.sum(v * v)) / (1 - np.sum(w * w))
            arcosh = lambda x: np.log(x + np.sqrt((x + 1) * (x - 1)))
            return arcosh(1 + delta)
        return umap.UMAP(metric=dist_on_poincare_disk)
    raise ValueError('The metric should be "euclidean", "cosine", or "dist_on_poincare_disk".')

def umap_embedding(text_vectors, keys, metric, save_plain_embed_path, save_umap_model_path):
    """
    Calculate a UMAP plain embedding.
    """
    logging.info('Start calculating a UMAP plain embedding.')

    model = get_umap_model(metric)
    plain_embeds = model.fit_transform(text_vectors)

    with open(save_plain_embed_path, 'wb') as _:
        dill.dump((keys, plain_embeds), _)

    with open(save_umap_model_path, 'wb') as _:
        dill.dump(model, _)

    logging.info('Finish calculating a plain embedding.')
