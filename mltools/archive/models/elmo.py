"""
Define an ELMo model.
"""
import numpy as np
import torch
import torch.nn as nn

from nlp_model.utils import pad_texts
from nlp_model.interfaces import TextEmbedder

class ELMo(TextEmbedder, nn.Module):
    """
    Define an ELMo pretraining model.
    """
    def __init__(
            self, text_processor, emb_dim, rnn_layer_count, forward_weights, backward_weights, use_cuda=False):
        super(ELMo, self).__init__()

        self.text_processor = text_processor
        self.use_cuda = use_cuda

        self.rnn_layer_count = rnn_layer_count

        self.weight_dict = nn.ParameterDict({
            'forward_dir': nn.Parameter(
                torch.Tensor(np.array(forward_weights))),
            'backward_dir': nn.Parameter(
                torch.Tensor(np.array(backward_weights)))
        })
        self.weight_dict['forward_dir'].requires_grad = True
        self.weight_dict['backward_dir'].requires_grad = True

        self.embed_layer = nn.Embedding(self.vocab_count, emb_dim, padding_idx=self.pad_index)

        self.lstms = nn.ModuleList([
            nn.LSTM(
                self.emb_dim, self.emb_dim,
                num_layers=1, batch_first=True, bidirectional=False)
            for _ in range(self.rnn_layer_count)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.emb_dim) for _ in range(self.rnn_layer_count)
        ])

        self.output_layer = nn.Linear(self.emb_dim, self.vocab_count)

        self.activate = nn.LogSoftmax(dim=2)
        self.loss = nn.NLLLoss(ignore_index=self.pad_index, reduction='none')

        if self.use_cuda:
            self.cuda()

    @property
    def vocab_count(self):
        return self.text_processor.vocab_count

    @property
    def pad_index(self):
        return self.text_processor.pad_index

    @property
    def emb_dim(self):
        return self.embed_layer.embedding_dim

    def tokenize(self, text):
        return self.text_processor.tokenize(text)

    def index_tokens(self, tokens):
        return self.text_processor.index_tokens(tokens)

    def forward(self, texts):
        curr = self.embed_layer(texts)

        for lstm, layer_norm in zip(self.lstms, self.layer_norms):
            prev_curr = curr
            curr, _ = lstm(prev_curr)
            curr = prev_curr + curr
            curr = layer_norm(curr)

        return curr

    def fit(self, texts):
        self.train()

        # Preprocess Data
        padded_texts, masks, _ = pad_texts(texts, self.pad_index)

        padded_text_tensor = torch.LongTensor(padded_texts)
        input_mask_tensor = torch.Tensor(masks[:, :-1])
        if self.use_cuda:
            padded_text_tensor = padded_text_tensor.cuda()
            input_mask_tensor = input_mask_tensor.cuda()

        input_text_tensor = padded_text_tensor[:, :-1]
        output_text_tensor = padded_text_tensor[:, 1:]

        curr = self(input_text_tensor)
        curr = self.output_layer(curr)
        outputs = self.activate(curr)

        loss = self.loss(
            outputs.view(-1, outputs.shape[2]), output_text_tensor.reshape(-1))
        loss = loss.view(output_text_tensor.shape[0], -1) * (1 - input_mask_tensor)

        def reduction(seq_loss, mask_tensor):
            lengths = torch.sum(1 - mask_tensor, dim=0, keepdim=True)
            return torch.mean(torch.sum(seq_loss / lengths, dim=0))

        reduction_loss = reduction(loss, input_mask_tensor)
        if self.use_cuda:
            reduction_loss = reduction_loss.cpu()

        return reduction_loss

    def embed(self, texts):
        """
        両方向の Embedding 層、LSTM 層の出力結果を返します。
        """
        self.eval()

        # Preprocess Data
        padded_texts, masks, _ = pad_texts(texts, self.pad_index)

        input_text_tensor = torch.LongTensor(padded_texts)
        if self.use_cuda:
            input_text_tensor = input_text_tensor.cuda()

        embeds = self(input_text_tensor)
        if self.use_cuda:
            embeds = embeds.cpu()

        return embeds

    def before_serialize(self):
        params = {
            "use_cuda": self.use_cuda
        }

        self.text_processor.before_serialize()
        self.use_cuda = False
        self.cpu()

        return params

    def after_deserialize(self, use_cuda=False):
        self.text_processor.after_deserialize()
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.cuda()
