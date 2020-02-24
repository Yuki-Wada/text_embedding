"""
Do text clustering by Gaussian Mixture model.
"""
import logging
import dill

from sklearn.mixture import GaussianMixture

from nlp_model.utils import save_sjis_csv

def gaussian_mixture(
        text_vectors, keys, save_clustering_result_path, save_model_path,
        gaussian_mixture_params):
    """
    Do text clustering by Gaussian Mixture model.
    """
    logging.info('Start text clustering by Gaussian Mixture model.')

    model = GaussianMixture(**gaussian_mixture_params)
    model.fit(text_vectors)
    keys['ClusterID'] = model.predict(text_vectors) + 1

    save_sjis_csv(keys, save_clustering_result_path)
    with open(save_model_path, 'wb') as _:
        dill.dump(model, _)

    logging.info('Finish text clustering.')
