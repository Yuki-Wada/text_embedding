"""
Unit Test
"""
import unittest
import numpy as np
import torch

from mltools.model.word2vec_impl.word2vec_impl_cython \
    import update_w_cython, update_w_naive, update_w_eigen, update_w_avx # pylint: disable=import-error,no-name-in-module

class TestStringMethods(unittest.TestCase):
    def test_word2vec_impl(self):
        np.random.seed()

        vocab_count = 10000
        hidden_dim = 500
        batch_size = 1000

        w_in_original = np.random.randn(hidden_dim, vocab_count).astype(np.float32)
        w_out_original = np.random.randn(vocab_count, hidden_dim).astype(np.float32)

        indices_in = np.random.randint(0, vocab_count, batch_size)
        indices_out = np.random.randint(0, vocab_count, batch_size)
        labels = np.random.randint(0, 2, batch_size)
        lr = 1e-1

        w_in_cython = w_in_original.copy()
        w_out_cython = w_out_original.copy()
        update_w_cython(w_in_cython, w_out_cython, indices_in, indices_out, labels, lr)

        w_in_naive = w_in_original.transpose(1, 0).copy()
        w_out_naive = w_out_original.copy()
        update_w_naive(w_in_naive, w_out_naive, indices_in, indices_out, labels, lr)
        w_in_naive = w_in_naive.transpose(1, 0)

        w_in_eigen = w_in_original.transpose(1, 0).copy()
        w_out_eigen = w_out_original.copy()
        update_w_eigen(w_in_eigen, w_out_eigen, indices_in, indices_out, labels, lr)
        w_in_eigen = w_in_eigen.transpose(1, 0)

        w_in_avx = w_in_original.transpose(1, 0).copy()
        w_out_avx = w_out_original.copy()
        update_w_avx(w_in_avx, w_out_avx, indices_in, indices_out, labels, lr)
        w_in_avx = w_in_avx.transpose(1, 0)

        w_in_torch = torch.tensor(w_in_original.transpose(1, 0), requires_grad=True)
        w_out_torch = torch.tensor(w_out_original, requires_grad=True)
        sgd = torch.optim.SGD([w_in_torch, w_out_torch], lr=lr)
        for i, (index_in, index_out, label) in enumerate(zip(indices_in, indices_out, labels)):
            sgd.zero_grad()
            outputs = torch.sum(w_in_torch[index_in] * w_out_torch[index_out])
            loss = torch.nn.BCEWithLogitsLoss(reduction='sum')(outputs, torch.tensor(label, dtype=torch.float))
            loss.backward()
            sgd.step()

        w_in_torch = w_in_torch.data.numpy()
        w_out_torch = w_out_torch.data.numpy()
        w_in_torch = w_in_torch.transpose(1, 0)

        self.assertLess(np.mean(np.abs(w_in_cython - w_in_naive)), 1e-6)
        self.assertLess(np.mean(np.abs(w_out_cython - w_out_naive)), 1e-6)
        self.assertLess(np.mean(np.abs(w_in_cython - w_in_eigen)), 1e-6)
        self.assertLess(np.mean(np.abs(w_out_cython - w_out_eigen)), 1e-6)
        self.assertLess(np.mean(np.abs(w_in_cython - w_in_avx)), 1e-6)
        self.assertLess(np.mean(np.abs(w_out_cython - w_out_avx)), 1e-6)
        self.assertLess(np.mean(np.abs(w_in_cython - w_in_torch)), 1e-6)
        self.assertLess(np.mean(np.abs(w_out_cython - w_out_torch)), 1e-6)

if __name__ == '__main__':
    unittest.main()
