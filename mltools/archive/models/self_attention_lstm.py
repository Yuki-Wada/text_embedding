"""
Define an LSTM model with self-attention mechanism which assign a label to an input text.
"""
import dill

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

from nlp_model.constant import INF
from nlp_model.utils import get_embed_layer, pad_texts

class SelfAttentionLayer(nn.Module):
    def __init__(self, config):
        super(SelfAttentionLayer, self).__init__()

        input_dim = config['input_dim']
        hidden_dim = config['hidden_dim']
        attention_dim = config['attention_dim']

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activate1 = nn.Tanh()
        self.linear2 = nn.Linear(hidden_dim, attention_dim)
        self.activate2 = nn.Softmax(dim=1)

    def forward(self, inputs, mask):
        batch_count = inputs.shape[0]

        curr = self.linear1(inputs)
        curr = self.activate1(curr)
        curr = self.linear2(curr)
        curr -= torch.unsqueeze(mask * INF, 2)
        attention = self.activate2(curr)

        result = torch.matmul(torch.transpose(inputs, 1, 2), attention).view(batch_count, -1)

        return result, attention

class AttentionLSTMLayer(nn.Module):
    """
    Define the LSTM model which assigns a label to an input text.
    """

    def __init__(
            self, text_processor, label_converter,
            embed_params, rnn_layer_params, self_attention_params,
            output_dim, use_cuda=False):
        super(AttentionLSTMLayer, self).__init__()

        self.text_processor = text_processor
        self.label_converter = label_converter
        self.use_cuda = use_cuda

        embed_params['args']['vocab_count'] = self.text_processor.vocab_count
        self.embed_layer, self.emb_dim = get_embed_layer(**embed_params)
        self.hidden_dim = rnn_layer_params['hidden_size']

        rnn_layer_params['input_size'] = self.emb_dim
        rnn_layer_params['batch_first'] = True
        self.rnn_layer = nn.LSTM(**rnn_layer_params)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)

        self_attention_params['input_dim'] = self.hidden_dim
        self.attention_layer = SelfAttentionLayer(self_attention_params)
        self.output_layer = nn.Linear(
            self.hidden_dim * self_attention_params['attention_dim'], output_dim)

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
        embeds, masks, lengths = input_data
        curr = rnn.pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
        curr, _ = self.rnn_layer(curr)
        curr, _ = rnn.pad_packed_sequence(curr, batch_first=True)
        curr = curr[:, :, :self.hidden_dim] + curr[:, :, self.hidden_dim:]

        curr = self.layer_norm(curr)

        curr *= torch.unsqueeze(1 - masks, 2)
        curr, attentions = self.attention_layer(curr, masks)
        result = self.output_layer(curr)

        return result, attentions

    def fit(self, texts, labels):
        """
        Train the model by using pairs of a text and a label.
        """
        self.train()

        padded_texts, masks, lengths = pad_texts(texts, self.pad_index)
        input_texts = torch.LongTensor(padded_texts)
        masks = torch.Tensor(masks)
        lengths = torch.LongTensor(lengths)
        labels = torch.LongTensor(labels)
        if self.use_cuda:
            input_texts = input_texts.cuda()
            masks = masks.cuda()
            lengths = lengths.cuda()
            labels = labels.cuda()

        embeds = self.embed_layer.embed(input_texts)
        results, _ = self(embeds, masks, lengths)
        probas = self.train_activate(results)
        loss = self.loss(probas, labels)

        if self.use_cuda:
            loss = loss.cpu()

        return loss

    def predict(self, texts):
        """
        Assign a label to an input text.
        """
        self.eval()

        padded_texts, masks, lengths = pad_texts(texts, self.pad_index)
        input_texts = torch.LongTensor(padded_texts)
        masks = torch.Tensor(masks)
        lengths = torch.LongTensor(lengths)
        if self.use_cuda:
            input_texts = input_texts.cuda()
            masks = masks.cuda()
            lengths = lengths.cuda()

        embeds = self.embed_layer(input_texts)
        results, attentions = self(embeds, masks, lengths)
        probas = self.predict_activate(results)

        if self.use_cuda:
            probas = probas.cpu()

        return probas, attentions

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
        self.use_cuda = False
        self.cpu()

    def after_deserialize(self, use_cuda=False):
        """
        You must execute this function after deserializing into this model.
        """
        self.text_processor.after_deserialize()
        self.label_converter.after_deserialize()
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
        assert isinstance(model, AttentionLSTMLayer)
        model.after_deserialize(use_cuda=use_cuda)
        return model
