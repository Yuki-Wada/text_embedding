"""
Define a transformer model which assign a label to an input text.
"""
import torch
import torch.nn as nn

from nlp_model.constant import INF
from nlp_model.utils import pad_texts, get_embed_layer
from nlp_model.interfaces import TextEmbedder, TextClassifier
from nlp_model.models.common_models import PositionalEncoding, SelfAttentionLayer

class GPT(TextEmbedder, nn.Module):
    """
    Define a pretraining GPT model.
    """
    def __init__(
            self, text_processor, embed_params, transformer_params, use_cuda=False):
        super(GPT, self).__init__()

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

        self.output_layer = nn.Linear(self.emb_dim, self.vocab_count)
        self.loss = nn.NLLLoss(ignore_index=self.pad_index, reduction='none')

        self.activate = nn.LogSoftmax(dim=2)

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
        texts, src_masks, src_key_padding_masks = input_data
        seq_len = texts.shape[0]
        curr = self.embed_layer(texts)
        pos_enc = self.positional_encoding.encode(seq_len)
        curr += torch.unsqueeze(pos_enc, 1)

        outputs = self.transformer_encoder(
            curr, mask=src_masks, src_key_padding_mask=src_key_padding_masks)

        return outputs

    def pretrain(self, texts):
        """
        Pretrain this model.
        """
        self.train()

        # Preprocess Data
        padded_texts, masks, _ = pad_texts(texts, self.pad_index)
        padded_texts = padded_texts.transpose(1, 0)

        padded_text_tensor = torch.LongTensor(padded_texts)
        mask_tensor = torch.Tensor(masks.transpose(1, 0))

        seq_len = padded_text_tensor[:-1].shape[0]

        input_mask_tensor = mask_tensor[:-1]
        src_mask_tensor = - torch.triu(torch.ones(seq_len, seq_len), diagonal=1) * INF
        src_key_padding_mask_tensor = torch.BoolTensor(
            mask_tensor[:-1].transpose(1, 0) == 1.0)
        if self.use_cuda:
            padded_text_tensor = padded_text_tensor.cuda()
            input_mask_tensor = input_mask_tensor.cuda()
            src_mask_tensor = src_mask_tensor.cuda()
            src_key_padding_mask_tensor = src_key_padding_mask_tensor.cuda()

        input_text_tensor = padded_text_tensor[:-1]
        output_text_tensor = padded_text_tensor[1:]

        # Input Data
        embeds = self(input_text_tensor, src_mask_tensor, src_key_padding_mask_tensor)
        outputs = self.output_layer(embeds)
        probas = self.activate(outputs)

        # Calculate Loss
        loss = self.loss(
            probas.view(-1, probas.shape[2]), output_text_tensor.reshape(-1))
        loss = loss.view(-1, padded_texts.shape[1]) * (1 - input_mask_tensor)

        def reduction(seq_loss, mask_tensor):
            lengths = torch.sum(1 - mask_tensor, dim=0, keepdim=True)
            return torch.mean(torch.sum(seq_loss / lengths, dim=0))

        reduction_loss = reduction(loss, input_mask_tensor)
        if self.use_cuda:
            reduction_loss = reduction_loss.cpu()

        return reduction_loss

    def embed(self, texts, batch_first=True):
        """
        Assign a label to an input text.
        """
        self.eval()

        padded_texts, masks, _ = pad_texts(texts, self.pad_index)
        padded_text_tensor = torch.LongTensor(padded_texts.transpose(1, 0))
        mask_tensor = torch.Tensor(masks.transpose(1, 0))

        seq_len = padded_text_tensor.shape[0]

        src_mask_tensor = - torch.triu(torch.ones(seq_len, seq_len), diagonal=1) * INF
        src_key_padding_mask_tensor = torch.BoolTensor(
            mask_tensor.transpose(1, 0) == 1.0)
        if self.use_cuda:
            padded_text_tensor = padded_text_tensor.cuda()
            mask_tensor = mask_tensor.cuda()
            src_mask_tensor = src_mask_tensor.cuda()
            src_key_padding_mask_tensor = src_key_padding_mask_tensor.cuda()

        embeds = self(
            padded_text_tensor, src_mask_tensor, src_key_padding_mask_tensor)

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
        self.positional_encoding.after_deserialize(use_cuda=use_cuda)
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.cuda()

class GPTFineTuner(TextClassifier, nn.Module):
    """
    Define a GPT finetuning model which assigns a label to an input text.
    """
    def __init__(
            self, gpt_model, label_converter, self_attention_params,
            output_dim, pretraining_coefficient, use_cuda=False):
        super(GPTFineTuner, self).__init__()

        self.label_converter = label_converter
        self.use_cuda = use_cuda

        self.gpt_model = gpt_model
        self.attention_layer = SelfAttentionLayer(
            self.emb_dim, use_cuda=self.use_cuda, **self_attention_params)
        self.output_layer = nn.Linear(
            self.emb_dim * self_attention_params['attention_dim'], output_dim)
        self.train_activate = nn.LogSoftmax(dim=1)
        self.predict_activate = nn.Softmax(dim=1)
        self.loss = nn.NLLLoss()

        self.pretraining_coefficient = pretraining_coefficient

        if self.use_cuda:
            self.cuda()

    @property
    def vocab_count(self):
        return self.gpt_model.vocab_count

    @property
    def pad_index(self):
        return self.gpt_model.pad_index

    @property
    def emb_dim(self):
        return self.gpt_model.emb_dim

    @property
    def text_processor(self):
        """
        Return the text processor used in this model.
        """
        return self.gpt_model.text_processor

    def tokenize(self, text):
        return self.gpt_model.tokenize(text)

    def index_tokens(self, tokens):
        return self.gpt_model.index_tokens(tokens)

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

        seq_len = padded_text_tensor[:-1].shape[0]

        input_mask_tensor = mask_tensor[:-1]
        src_mask_tensor = - torch.triu(torch.ones(seq_len, seq_len), diagonal=1) * INF
        src_key_padding_mask_tensor = torch.BoolTensor(
            mask_tensor[:-1].transpose(1, 0) == 1.0)
        label_tensor = torch.LongTensor(labels)
        if self.use_cuda:
            padded_text_tensor = padded_text_tensor.cuda()
            input_mask_tensor = input_mask_tensor.cuda()
            src_mask_tensor = src_mask_tensor.cuda()
            src_key_padding_mask_tensor = src_key_padding_mask_tensor.cuda()
            label_tensor = label_tensor.cuda()

        input_text_tensor = padded_text_tensor[:-1]
        output_text_tensor = padded_text_tensor[1:]

        # Input Data
        embeds = self.gpt_model(
            input_text_tensor, src_mask_tensor, src_key_padding_mask_tensor)
        outputs = self.gpt_model.output_layer(embeds)
        probas = self.gpt_model.activate(outputs)

        weighted_vectors, _ = self.attention_layer(
            embeds.transpose(1, 0), input_mask_tensor.transpose(1, 0))
        outputs = self.output_layer(weighted_vectors)
        log_probas = self.train_activate(outputs)

        # Calculate Loss
        pretraining_loss = self.gpt_model.loss(
            probas.view(-1, probas.shape[2]), output_text_tensor.reshape(-1))
        pretraining_loss = pretraining_loss.view(-1, padded_texts.shape[1]) * \
            (1 - input_mask_tensor)

        def reduction(seq_loss, mask_tensor):
            lengths = torch.sum(1 - mask_tensor, dim=0, keepdim=True)
            return torch.mean(torch.sum(seq_loss / lengths, dim=0))

        pretraining_loss = reduction(pretraining_loss, input_mask_tensor)

        finetuning_loss = self.loss(log_probas, label_tensor)

        loss = finetuning_loss + pretraining_loss * self.pretraining_coefficient

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

        seq_len = padded_text_tensor.shape[0]

        src_mask_tensor = - torch.triu(torch.ones(seq_len, seq_len), diagonal=1) * INF
        src_key_padding_mask_tensor = torch.BoolTensor(
            mask_tensor.transpose(1, 0) == 1.0)
        if self.use_cuda:
            padded_text_tensor = padded_text_tensor.cuda()
            mask_tensor = mask_tensor.cuda()
            src_mask_tensor = src_mask_tensor.cuda()
            src_key_padding_mask_tensor = src_key_padding_mask_tensor.cuda()

        embeds = self.gpt_model(
            padded_text_tensor, src_mask_tensor, src_key_padding_mask_tensor)
        weighted_vectors, _ = self.attention_layer(
            embeds.transpose(1, 0), mask_tensor.transpose(1, 0))
        outputs = self.output_layer(weighted_vectors)
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

        self.gpt_model.before_serialize()
        self.label_converter.before_serialize()
        self.use_cuda = False
        self.cpu()

        return params

    def after_deserialize(self, use_cuda=False):
        self.gpt_model.after_deserialize()
        self.label_converter.after_deserialize()
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.cuda()