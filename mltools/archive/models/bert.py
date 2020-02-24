"""
Define a transformer model which assign a label to an input text.
"""
import numpy as np

import torch
import torch.nn as nn

from nlp_model.constant import EPSILON
from nlp_model.utils import pad_texts, get_embed_layer
from nlp_model.interfaces import TextEmbedder, TextClassifier
from nlp_model.models.common_models import PositionalEncoding

class BERT(TextEmbedder, nn.Module):
    """
    Define a BERT pretraining model.
    """
    def __init__(
            self, text_processor, embed_params, transformer_params, use_cuda=False):
        super(BERT, self).__init__()

        self.text_processor = text_processor
        self.use_cuda = use_cuda

        embed_params['args']['vocab_count'] = self.text_processor.vocab_count
        self.embed_layer, _ = get_embed_layer(**embed_params)
        self.positional_encoding = PositionalEncoding(self.emb_dim, self.use_cuda)

        transformer_encoder = nn.TransformerEncoderLayer(
            self.emb_dim,
            transformer_params['head_count'],
            transformer_params['feedforward_hidden_dim'])
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_encoder, transformer_params['stack_count'])

        self.token_output_layer = nn.Linear(self.emb_dim, self.vocab_count)
        self.token_activate = nn.LogSoftmax(dim=2)
        self.token_loss = nn.NLLLoss(ignore_index=self.pad_index, reduction='none')

        self.label_output_layer = nn.Linear(self.emb_dim, 2)
        self.label_activate = nn.LogSoftmax(dim=1)
        self.label_loss = nn.NLLLoss()

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

    def forward(self, *input_data):
        """
        Return an embedding expression of an input text.
        """
        texts, src_key_padding_masks = input_data
        seq_len = texts.shape[0]
        curr = self.embed_layer(texts)
        pos_enc = self.positional_encoding.encode(seq_len)
        curr += torch.unsqueeze(pos_enc, 1)

        outputs = self.transformer_encoder(curr, src_key_padding_mask=src_key_padding_masks)

        return outputs

    def replace_texts(
            self, padded_texts, masks, do_transpose=True,
            do_replace_proba=0.15, replace_mask_proba=0.8, replace_random_proba=0.1):
        """
        文章をパディングすることで、それぞれ長さの異なる文章群を 2 次元の numpy.ndarray の形にします。
        文章の長さ、パディング領域のマスクも同時に返します。
        """
        replaced_texts = np.copy(padded_texts)
        have_replaced = (np.random.rand(*padded_texts.shape) < do_replace_proba) * \
            (padded_texts != self.text_processor.cls_index) * \
            (padded_texts != self.text_processor.sep_index) * \
            (masks == 0)

        random_values = np.random.rand(*padded_texts.shape)
        replace_mask = have_replaced * (random_values < replace_mask_proba)
        replace_random = have_replaced * (replace_mask_proba <= random_values) * \
            (random_values < random_values + replace_random_proba)

        replaced_texts[replace_mask] = self.text_processor.mask_index
        replaced_texts[replace_random] = \
            np.random.randint(0, self.text_processor.vocab_count, np.sum(replace_random))

        if do_transpose:
            replaced_texts = replaced_texts.transpose(1, 0)
            have_replaced = have_replaced.transpose(1, 0)

        return replaced_texts, have_replaced

    def pretrain(self, first_sentences, second_sentences, labels):
        """
        Pretrain this model.
        """
        self.train()

        # Preprocess Data
        texts = [
            [self.text_processor.cls_index] + first_sentence + \
            [self.text_processor.sep_index] + second_sentence
            for first_sentence, second_sentence in zip(first_sentences, second_sentences)]

        padded_texts, masks, _ = pad_texts(texts)
        replaced_texts, have_replaced = self.replace_texts(padded_texts, masks)
        padded_texts = padded_texts.transpose(1, 0)

        input_text_tensor = torch.LongTensor(replaced_texts)
        mask_tensor = torch.Tensor(masks.transpose(1, 0))
        output_text_tensor = torch.LongTensor(padded_texts)
        label_tensor = torch.LongTensor(labels)
        have_replaced_tensor = torch.Tensor(have_replaced)

        src_key_padding_mask_tensor = torch.BoolTensor(
            mask_tensor.transpose(1, 0) == 1.0)
        if self.use_cuda:
            input_text_tensor = input_text_tensor.cuda()
            mask_tensor = mask_tensor.cuda()
            src_key_padding_mask_tensor = src_key_padding_mask_tensor.cuda()
            output_text_tensor = output_text_tensor.cuda()
            label_tensor = label_tensor.cuda()
            have_replaced_tensor = have_replaced_tensor.cuda()

        # Input Data
        embeds = self(input_text_tensor, src_key_padding_mask_tensor)
        outputs = self.token_output_layer(embeds)
        probas = self.token_activate(outputs)

        # Calculate Loss
        token_loss = self.token_loss(
            probas.view(-1, probas.shape[2]), output_text_tensor.reshape(-1))
        token_loss = token_loss.view(-1, input_text_tensor.shape[1]) * have_replaced_tensor
        token_loss = torch.sum(token_loss) / (torch.sum(have_replaced_tensor) + EPSILON)

        bbb = self.label_output_layer(embeds[0])
        ccc = self.label_activate(bbb)
        ddd = self.label_loss(ccc, label_tensor)

        total_loss = token_loss + ddd
        if self.use_cuda:
            total_loss = total_loss.cpu()

        return total_loss

    def embed(self, texts, batch_first=True):
        """
        Pretrain this model.
        """
        self.eval()

        padded_texts, masks, _ = pad_texts(texts, self.pad_index)
        padded_text_tensor = torch.LongTensor(padded_texts.transpose(1, 0))
        mask_tensor = torch.Tensor(masks.transpose(1, 0))

        src_key_padding_mask_tensor = torch.BoolTensor(
            mask_tensor.transpose(1, 0) == 1.0)
        if self.use_cuda:
            padded_text_tensor = padded_text_tensor.cuda()
            mask_tensor = mask_tensor.cuda()
            src_key_padding_mask_tensor = src_key_padding_mask_tensor.cuda()

        # Input Data
        embeds = self(padded_text_tensor, src_key_padding_mask_tensor)

        if batch_first:
            embeds = embeds.transpose(1, 0)

        if self.use_cuda:
            embeds = embeds.cpu()

        return embeds

    def before_serialize(self):
        params = {
            "use_cuda": self.use_cuda
        }

        self.text_processor.before_serialize()
        self.positional_encoding.before_serialize()
        self.use_cuda = False
        self.cpu()

        return params

    def after_deserialize(self, use_cuda=False):
        self.text_processor.after_deserialize()
        self.positional_encoding.after_deserialize(use_cuda)
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.cuda()

class BERTFineTuner(TextClassifier, nn.Module):
    """
    Define a BERT finetuning model which assigns a label to an input text.
    """
    def __init__(
            self, bert_model, label_converter, output_dim, use_cuda=False):
        super(BERTFineTuner, self).__init__()

        self.label_converter = label_converter
        self.use_cuda = use_cuda

        self.bert_model = bert_model

        self.output_layer = nn.Linear(self.emb_dim, output_dim)

        self.train_activate = nn.LogSoftmax(dim=1)
        self.predict_activate = nn.Softmax(dim=1)
        self.loss = nn.NLLLoss()

        if self.use_cuda:
            self.cuda()

    @property
    def vocab_count(self):
        return self.bert_model.vocab_count

    @property
    def pad_index(self):
        return self.bert_model.pad_index

    @property
    def emb_dim(self):
        return self.bert_model.emb_dim

    @property
    def text_processor(self):
        """
        Return the text processor used in this model.
        """
        return self.bert_model.text_processor

    def tokenize(self, text):
        return self.bert_model.tokenize(text)

    def index_tokens(self, tokens):
        return self.bert_model.index_tokens(tokens)

    def finetune(self, texts, labels):
        """
        Finetune this model.
        """
        self.train()

        # Preprocess Data
        padded_texts, masks, _ = pad_texts(texts, self.pad_index)
        padded_texts = padded_texts.transpose(1, 0)

        padded_text_tensor = torch.LongTensor(padded_texts)
        mask_tensor = torch.Tensor(masks.transpose(1, 0))

        src_key_padding_mask_tensor = torch.BoolTensor(
            mask_tensor.transpose(1, 0) == 1.0)
        label_tensor = torch.LongTensor(labels)
        if self.use_cuda:
            padded_text_tensor = padded_text_tensor.cuda()
            mask_tensor = mask_tensor.cuda()
            src_key_padding_mask_tensor = src_key_padding_mask_tensor.cuda()
            label_tensor = label_tensor.cuda()

        # Input Data
        embeds = self.bert_model(padded_text_tensor, src_key_padding_mask_tensor)
        outputs = self.output_layer(embeds[0])
        log_probas = self.train_activate(outputs)

        # Calculate Loss
        loss = self.loss(log_probas, label_tensor)

        if self.use_cuda:
            loss = loss.cpu()

        return loss

    def predict(self, texts):
        """
        Assign a label to an input text.
        """
        self.eval()

        padded_texts, masks, _ = pad_texts(texts, self.pad_index)
        padded_text_tensor = torch.LongTensor(padded_texts.transpose(1, 0))
        mask_tensor = torch.Tensor(masks.transpose(1, 0))

        src_key_padding_mask_tensor = torch.BoolTensor(
            mask_tensor.transpose(1, 0) == 1.0)
        if self.use_cuda:
            padded_text_tensor = padded_text_tensor.cuda()
            src_key_padding_mask_tensor = src_key_padding_mask_tensor.cuda()

        embeds = self.bert_model(padded_text_tensor, src_key_padding_mask_tensor)
        outputs = self.output_layer(embeds[0])
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
        params = {
            "use_cuda": self.use_cuda
        }

        self.bert_model.before_serialize()
        self.label_converter.before_serialize()
        self.use_cuda = False
        self.cpu()

        return params

    def after_deserialize(self, use_cuda=False):
        self.bert_model.after_deserialize()
        self.label_converter.after_deserialize()
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.cuda()
