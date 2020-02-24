"""
Define a transformer model which assign a label to an input text.
"""
import dill

import torch
import torch.nn as nn

from nlp_model.utils import pad_texts, get_embed_layer
from nlp_model.models.positional_encoding import PositionalEncoding

class TransformerEncoder(nn.Module):
    """
    Define a transformer model which assigns a label to an input text.
    """
    def __init__(
            self, text_processor, label_converter, embed_params, transformer_params,
            output_dim, use_cuda=False):
        super(TransformerEncoder, self).__init__()
        self.text_processor = text_processor
        self.label_converter = label_converter
        self.use_cuda = use_cuda

        embed_params['args']['vocab_count'] = self.text_processor.vocab_count
        self.embed_layer, self.emb_dim = get_embed_layer(**embed_params)
        self.positional_encoding = PositionalEncoding(self.emb_dim, self.use_cuda)

        transformer_encoder = nn.TransformerEncoderLayer(
            self.emb_dim,
            transformer_params['head_count'],
            transformer_params['feedforward_hidden_dim'])
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_encoder, transformer_params['stack_count'])

        self.output_layer = nn.Linear(self.emb_dim, output_dim)

        self.train_activate = nn.LogSoftmax(dim=1)
        self.predict_activate = nn.Softmax(dim=1)
        self.loss = nn.NLLLoss()

        if self.use_cuda:
            self.cuda()

    @property
    def pad_index(self):
        """
        Return a PAD token's index used in this model.
        """
        return self.text_processor.pad_index

    def forward(self, *input_data):
        texts, masks = input_data

        seq_len = texts.shape[0]
        curr = self.embed_layer(texts)
        pos_enc = self.positional_encoding.encode(seq_len)
        curr += torch.unsqueeze(pos_enc, 1)

        curr = self.transformer_encoder(curr, src_key_padding_mask=masks)
        outputs = self.output_layer(curr[0, :, :])

        return outputs

    def fit(self, texts, labels):
        """
        Train this model by inputing pairs of a text and a label.
        """
        self.train()

        padded_texts, masks, _ = pad_texts(texts, self.pad_index)
        padded_texts = torch.LongTensor(padded_texts.transpose(1, 0))
        masks = torch.BoolTensor(masks == 1)
        labels = torch.LongTensor(labels)
        if self.use_cuda:
            padded_texts = padded_texts.cuda()
            masks = masks.cuda()
            labels = labels.cuda()

        outputs = self(padded_texts, masks)
        probas = self.train_activate(outputs)
        loss = self.loss(probas, labels)

        if self.use_cuda:
            loss = loss.cpu()

        return loss

    def predict(self, texts):
        """
        Assign a label to an input text.
        """
        self.eval()

        padded_texts, masks, _ = pad_texts(texts, self.pad_index)
        padded_texts = torch.LongTensor(padded_texts.transpose(1, 0))
        masks = torch.BoolTensor(masks == 1)
        if self.use_cuda:
            padded_texts = padded_texts.cuda()
            masks = masks.cuda()

        outputs = self(padded_texts, masks)
        probas = self.predict_activate(outputs)

        if self.use_cuda:
            probas = probas.cpu()

        return probas

    def get_loss(self, probas, labels):
        """
        Calculate the same loss as training this model.
        """
        return self.loss(torch.log(torch.Tensor(probas)), torch.LongTensor(labels))

    def before_serialize(self):
        """
        You must execute this function before serializing this model.
        """
        self.text_processor.before_serialize()
        self.label_converter.before_serialize()
        self.positional_encoding.before_serialize()
        self.use_cuda = False
        self.cpu()

    def after_deserialize(self, use_cuda=False):
        """
        You must execute this function after deserializing into this model.
        """
        self.text_processor.after_deserialize()
        self.label_converter.after_deserialize()
        self.positional_encoding.after_deserialize(use_cuda=use_cuda)
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.cuda()

    def save(self, save_path):
        """
        Save the model by serializing it to a pickled file.
        """
        use_cuda = self.use_cuda
        self.before_serialize()
        with open(save_path, 'wb') as _:
            dill.dump(self, _)
        self.after_deserialize(use_cuda=use_cuda)

    @staticmethod
    def load(load_path, use_cuda=False):
        """
        Load the model by deserializing a loaded pickled file.
        """
        with open(load_path, 'rb') as _:
            model = dill.load(_)
        assert isinstance(model, TransformerEncoder), 'TransformerEncoder モデルのファイルパスを指定してください。'
        model.after_deserialize(use_cuda=use_cuda)
        return model
