"""
Define Encoder-Decoder Pytorch models.
"""
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from mltools.model.attention_utils import PositionalEncoder, MultiHeadAttention, \
    TransformerEncoderBlock

logger = logging.getLogger(__name__)

def get_masks(inputs, pad_index=0):
    return inputs == pad_index

def execute_rnn(inputs, masks, rnn_layer, state=None):
    lengths = np.sum(1 - masks.cpu().data.numpy(), axis=0).tolist()
    packed = rnn_utils.pack_padded_sequence(inputs, lengths, enforce_sorted=False)

    lstm_outputs, state = rnn_layer(packed, state)

    outputs, _ = rnn_utils.pad_packed_sequence(lstm_outputs)

    if rnn_layer.bidirectional:
        hidden_dim = outputs.shape[2] // 2
        outputs = outputs[:, :, :hidden_dim] + outputs[:, :, hidden_dim:]
        h, c = state
        h = h[:1] + h[1:]
        c = c[:1] + c[1:]
        state = h, c

    return outputs, state

def decoder_loss(true, prob, pad_index=0):
    masks = get_masks(true, pad_index)

    losses = nn.NLLLoss(reduction='none')(
        nn.LogSoftmax(dim=1)(prob.view(-1, prob.shape[2])), true.reshape(-1))
    losses = losses.view(true.shape[0], -1) * (1 - masks.float())

    lengths = torch.sum((1 - masks.float()), dim=0)
    losses /= lengths + 1e-18

    mean_loss = torch.mean(torch.sum(losses, dim=0))

    return mean_loss

def calc_bleu_score(true, pred, max_n=4):
    def get_ngram_count_dict(tokens, n):
        ngram_count_dict = {}
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i : i + n])
            if ngram not in ngram_count_dict:
                ngram_count_dict[ngram] = 0
            ngram_count_dict[ngram] += 1

        return ngram_count_dict

    def get_modified_ngram_precision(true, pred, n):
        true_ngram_count_dict = get_ngram_count_dict(true, n)
        pred_ngram_count_dict = get_ngram_count_dict(pred, n)

        if not pred_ngram_count_dict:
            return 0

        clipped_count = 0
        total_pred_ngram_count = 0
        for ngram in pred_ngram_count_dict:
            true_count = true_ngram_count_dict[ngram] if ngram in true_ngram_count_dict else 0
            pred_count = pred_ngram_count_dict[ngram]
            total_pred_ngram_count += pred_count
            clipped_count += min(true_count, pred_count)

        return clipped_count / total_pred_ngram_count

    if not pred:
        return 0

    score = 1
    for n in range(1, max_n + 1):
        score *= get_modified_ngram_precision(true, pred, n)
    score **= 1 / max_n

    score *= np.exp(min(0, 1 - len(true) / len(pred)))
    score *= 100

    return score

def calc_bleu_scores(trues, preds, eos_index, max_n=4):
    scores = []
    for true_indices, pred_indices in zip(trues, preds):
        true = []
        pred = []
        for index in true_indices:
            if index == eos_index:
                break
            true.append(index)
        for index in pred_indices:
            if index == eos_index:
                break
            pred.append(index)

        scores.append(calc_bleu_score(true, pred, max_n=max_n))

    return scores

class GreedyChoiceDecoder:
    def __init__(self, mb_size, seq_len, begin_of_encode_index):
        self.mb_size = mb_size
        self.seq_len = seq_len
        self.begin_of_encode_index = begin_of_encode_index

        self.seq_index = 0

    def start(self, state):
        paths = np.zeros((self.mb_size, self.seq_len + 1), dtype=np.int)
        paths[:, 0] = self.begin_of_encode_index
        all_outputs = None
        probs = np.ones((self.mb_size,))

        return [(state, all_outputs, paths, probs)]

    def decode(self, output_candidates):
        self.seq_index += 1

        (decoder_outputs, state), all_outputs, paths, probs = output_candidates[0]
        decoder_probs = F.softmax(decoder_outputs, dim=1).cpu().data.numpy()

        for mb_idx in range(self.mb_size):
            decoder_probs[mb_idx, 0] = 0
            word_idx = np.argmax(decoder_probs[mb_idx])

            paths[mb_idx, self.seq_index] = word_idx
            probs[mb_idx] *= decoder_probs[mb_idx, word_idx]

        return [(state, all_outputs, paths, probs)]

class RandomChoiceDecoder:
    def __init__(self, mb_size, seq_len, begin_of_encode_index):
        self.mb_size = mb_size
        self.seq_len = seq_len
        self.begin_of_encode_index = begin_of_encode_index

        self.seq_index = 0

    def start(self, state):
        paths = np.zeros((self.mb_size, self.seq_len + 1), dtype=np.int)
        paths[:, 0] = self.begin_of_encode_index
        all_outputs = None
        probs = np.ones((self.mb_size,))

        return [(state, all_outputs, paths, probs)]

    def decode(self, output_candidates):
        self.seq_index += 1

        (decoder_outputs, state), all_outputs, paths, probs = output_candidates[0]
        decoder_probs = F.softmax(decoder_outputs, dim=1).cpu().data.numpy()
        decoder_probs = decoder_probs / np.sum(decoder_probs, axis=1, keepdims=True)

        for mb_idx in range(self.mb_size):
            word_idx = 0
            while word_idx == 0:
                word_idx = np.random.choice(decoder_probs.shape[1], p=decoder_probs[mb_idx])

            paths[mb_idx, self.seq_index] = word_idx
            probs[mb_idx] *= decoder_probs[mb_idx, word_idx]

        return [(state, all_outputs, paths, probs)]

class BeamSearchDecoder:
    def __init__(self, mb_size, seq_len, breadth_len, begin_of_encode_index):
        self.mb_size = mb_size
        self.seq_len = seq_len
        self.breadth_len = breadth_len
        self.begin_of_encode_index = begin_of_encode_index

        self.seq_index = 0

    def start(self, state):
        paths = np.zeros((self.mb_size, self.seq_len + 1), dtype=np.int)
        paths[:, 0] = self.begin_of_encode_index
        all_outputs = None
        probs = np.ones((self.mb_size,))

        return [(state, all_outputs, paths, probs)]

    def decode(self, output_candidates):
        self.seq_index += 1

        idx_count = len(output_candidates) * self.breadth_len
        prob_table = np.zeros((self.mb_size, idx_count))
        index_table = np.zeros((self.mb_size, idx_count)).astype(np.int32)
        for output_idx, ((decoder_outputs, _), _, _, base_probs) in enumerate(output_candidates):
            decoder_probs = F.softmax(decoder_outputs, dim=1).cpu().data.numpy()
            decoder_probs[:, 0] = 0
            decoder_probs = decoder_probs / np.sum(decoder_probs, axis=1, keepdims=True)
            indices = np.argsort(decoder_probs, axis=1)[:, ::-1]
            for breadth_idx in range(self.breadth_len):
                idx = output_idx * self.breadth_len + breadth_idx
                for mb_idx in range(self.mb_size):
                    prob_table[mb_idx, idx] = \
                        base_probs[mb_idx] * decoder_probs[mb_idx, indices[mb_idx, breadth_idx]]
                    index_table[mb_idx, idx] = indices[mb_idx, breadth_idx]

        selected_indices = np.argsort(prob_table, axis=1)[:, ::-1]

        next_input_candidates = []
        for idx in range(self.breadth_len):
            hs, cs = [], []
            all_outputs = []
            probs = np.zeros((self.mb_size,))
            paths = np.zeros((self.mb_size, self.seq_len + 1), dtype=np.int)

            for mb_idx in range(self.mb_size):
                selected_idx = selected_indices[mb_idx, idx]
                output_idx = int(selected_idx / self.breadth_len)

                state = output_candidates[output_idx][0][1]
                if state is not None:
                    h_n, c_n = state
                    hs.append(h_n[:, mb_idx:mb_idx+1])
                    cs.append(c_n[:, mb_idx:mb_idx+1])

                paths[mb_idx] = output_candidates[output_idx][2][mb_idx]
                all_outputs.append(output_candidates[output_idx][1][:, mb_idx:mb_idx+1])
                paths[mb_idx, self.seq_index] = index_table[mb_idx, selected_idx]

                probs[mb_idx] = prob_table[mb_idx, selected_idx]

            state = (torch.cat(hs, dim=1), torch.cat(cs, dim=1)) if hs else None
            all_outputs = torch.cat(all_outputs, dim=1)

            next_input_candidates.append((state, all_outputs, paths, probs))

        return next_input_candidates

def get_inference_decoder(decoding_params, mb_size):
    begin_of_encode_index = decoding_params['begin_of_encode_index']
    seq_len = decoding_params['seq_len']

    if decoding_params['decoding_type'] == 'greedy_choice':
        return GreedyChoiceDecoder(mb_size, seq_len, begin_of_encode_index)
    if decoding_params['decoding_type'] == 'random_choice':
        return RandomChoiceDecoder(mb_size, seq_len, begin_of_encode_index)
    if decoding_params['decoding_type'] == 'beam_search':
        breadth_len = decoding_params['breadth_len']
        return BeamSearchDecoder(mb_size, seq_len, breadth_len, begin_of_encode_index)
    raise ValueError(
        'The inference decoder {} is not supported.'.format(decoding_params['decoding_type']))

class NaiveSeq2Seq(nn.Module):
    def __init__(
            self,
            encoder_vocab_count,
            decoder_vocab_count,
            emb_dim,
            hidden_dim,
            bidirectional,
            gpu_id=-1,
        ):
        super(NaiveSeq2Seq, self).__init__()

        self.encoder_embedding_layer = nn.Embedding(
            encoder_vocab_count,
            emb_dim,
        )
        self.encoder_rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            bidirectional=bidirectional,
        )

        self.decoder_embedding_layer = nn.Embedding(
            decoder_vocab_count,
            emb_dim,
        )
        self.decoder_rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
        )
        self.output_layer = nn.Linear(hidden_dim, decoder_vocab_count)

        if gpu_id >= 0:
            self.device = torch.device('cuda:{}'.format(gpu_id))
        else:
            self.device = torch.device('cpu')
        self.to(self.device)

    def encode(self, encoder_inputs):
        encoder_embedded = self.encoder_embedding_layer(encoder_inputs)
        encoder_masks = get_masks(encoder_inputs)
        encoder_outputs, encoder_state = execute_rnn(
            encoder_embedded, encoder_masks, self.encoder_rnn
        )

        return encoder_outputs, encoder_state

    def decode(self, decoder_inputs, states):
        decoder_embedded = self.decoder_embedding_layer(decoder_inputs)
        decoder_masks = get_masks(decoder_inputs)
        decoder_sequences, decoder_state = execute_rnn(
            decoder_embedded, decoder_masks, self.decoder_rnn, states,
        )

        decoder_outputs = self.output_layer(decoder_sequences)

        return decoder_outputs, decoder_state

    def forward(self, encoder_inputs, decoder_inputs):
        _, state = self.encode(encoder_inputs)
        decoder_outputs, _ = self.decode(decoder_inputs, state)

        return decoder_outputs

    def inference(self, encoder_inputs, decoding_params):
        inference_decoder = get_inference_decoder(decoding_params, encoder_inputs.shape[1])
        seq_len = decoding_params['seq_len']

        _, state = self.encode(encoder_inputs)

        input_candidates = inference_decoder.start(state)
        for seq in range(seq_len):
            output_candidates = []
            for state, all_outputs, paths, probs in input_candidates:
                decoder_inputs = torch.LongTensor(
                    paths[:, seq:seq+1].transpose(1, 0)).to(self.device)
                decoder_outputs, state = self.decode(decoder_inputs, state)
                all_outputs = torch.cat(
                    ([all_outputs] if all_outputs is not None else []) + [decoder_outputs])
                output_candidates.append(((decoder_outputs[0], state), all_outputs, paths, probs))

            input_candidates = inference_decoder.decode(output_candidates)

        return input_candidates[0][1], input_candidates[0][2][:, 1:]

class Seq2SeqWithGlobalAttention(nn.Module):
    def __init__(
            self,
            encoder_vocab_count,
            decoder_vocab_count,
            emb_dim,
            hidden_dim,
            bidirectional,
            gpu_id=-1,
        ):
        super(Seq2SeqWithGlobalAttention, self).__init__()

        self.encoder_embedding_layer = nn.Embedding(
            encoder_vocab_count,
            emb_dim,
        )
        self.encoder_rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            bidirectional=bidirectional,
        )

        self.decoder_embedding_layer = nn.Embedding(
            decoder_vocab_count,
            emb_dim,
        )
        self.decoder_rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
        )

        self.global_attention_layer = nn.Linear(hidden_dim, hidden_dim)

        self.output_layer = nn.Linear(hidden_dim * 2, decoder_vocab_count)

        if gpu_id >= 0:
            self.device = torch.device('cuda:{}'.format(gpu_id))
        else:
            self.device = torch.device('cpu')
        self.to(self.device)

    def encode(self, encoder_inputs):
        encoder_embedded = self.encoder_embedding_layer(encoder_inputs)
        encoder_masks = get_masks(encoder_inputs)
        encoder_outputs, encoder_state = execute_rnn(
            encoder_embedded, encoder_masks, self.encoder_rnn
        )

        return encoder_outputs, encoder_state

    def decode(self, decoder_inputs, states, encoder_outputs, encoder_masks):
        decoder_embedded = self.decoder_embedding_layer(decoder_inputs)
        decoder_masks = get_masks(decoder_inputs)
        decoder_sequences, decoder_state = execute_rnn(
            decoder_embedded, decoder_masks, self.decoder_rnn, states,
        )

        scores = torch.einsum(
            'imk,jmk->imj',
            self.global_attention_layer(decoder_sequences),
            encoder_outputs,
        )
        scores -= encoder_masks.float().transpose(1, 0) * 1e18
        attentions = F.softmax(scores, dim=2)
        weighted = torch.einsum('imj,jmk->imk', attentions, encoder_outputs)
        concat = torch.cat([decoder_sequences, weighted], dim=2)

        decoder_outputs = self.output_layer(concat)

        return decoder_outputs, decoder_state

    def forward(self, encoder_inputs, decoder_inputs):
        encoder_outputs, state = self.encode(encoder_inputs)
        encoder_masks = get_masks(encoder_inputs)
        decoder_outputs, _ = self.decode(
            decoder_inputs, state, encoder_outputs, encoder_masks)

        return decoder_outputs

    def inference(self, encoder_inputs, decoding_params):
        inference_decoder = get_inference_decoder(decoding_params, encoder_inputs.shape[1])
        seq_len = decoding_params['seq_len']

        encoder_outputs, state = self.encode(encoder_inputs)
        encoder_masks = get_masks(encoder_inputs)

        input_candidates = inference_decoder.start(state)
        for seq in range(seq_len):
            output_candidates = []
            for state, all_outputs, paths, probs in input_candidates:
                decoder_inputs = torch.LongTensor(
                    paths[:, seq:seq+1].transpose(1, 0)).to(self.device)
                decoder_outputs, state = self.decode(
                    decoder_inputs, state, encoder_outputs, encoder_masks)
                all_outputs = torch.cat(
                    ([all_outputs] if all_outputs is not None else []) + [decoder_outputs])
                output_candidates.append(((decoder_outputs[0], state), all_outputs, paths, probs))

            input_candidates = inference_decoder.decode(output_candidates)

        return input_candidates[0][1], input_candidates[0][2][:, 1:]

class TransformerDecoderBlock(nn.Module):
    def __init__(
            self,
            model_dim,
            head_count,
            feed_forward_hidden_dim,
            gpu_id=-1,
        ):
        super(TransformerDecoderBlock, self).__init__()
        self.masked_attention = MultiHeadAttention(model_dim, head_count, forward_masked=True)
        self.layer_norm_after_masked_attention = nn.LayerNorm(model_dim)

        self.key_value_attention = MultiHeadAttention(model_dim, head_count)
        self.layer_norm_after_key_value_attention = nn.LayerNorm(model_dim)

        self.hidden_layer = nn.Linear(model_dim, feed_forward_hidden_dim)
        self.output_layer = nn.Linear(feed_forward_hidden_dim, model_dim)
        self.layer_norm_after_feed_forward = nn.LayerNorm(model_dim)

        if gpu_id >= 0:
            self.device = torch.device('cuda:{}'.format(gpu_id))
        else:
            self.device = torch.device('cpu')
        self.to(self.device)

    def forward(self, inputs, key_value, input_mask=None, key_value_mask=None):
        h = self.masked_attention(inputs, inputs, inputs, input_mask)
        h = h + inputs
        h0 = self.layer_norm_after_masked_attention(h)

        h = self.key_value_attention(h0, key_value, key_value, key_value_mask)
        h = h + h0
        h0 = self.layer_norm_after_key_value_attention(h)

        h = self.hidden_layer(h0)
        h = self.output_layer(h)
        h = h + h0
        h = self.layer_norm_after_feed_forward(h)

        return h

class TransformerEncoderDecoder(nn.Module):
    def __init__(
            self,
            encoder_vocab_count,
            decoder_vocab_count,
            model_dim,
            head_count,
            feed_forward_hidden_dim,
            encoder_block_count,
            decoder_block_count,
            gpu_id=-1,
        ):
        super(TransformerEncoderDecoder, self).__init__()

        self.positional_encoder = PositionalEncoder(model_dim)

        self.encoder_embedding_layer = nn.Embedding(
            encoder_vocab_count,
            model_dim,
        )
        self.encoder_blocks = nn.ModuleList([])
        for _ in range(encoder_block_count):
            self.encoder_blocks.append(TransformerEncoderBlock(
                model_dim, head_count, feed_forward_hidden_dim
            ))

        self.decoder_embedding_layer = nn.Embedding(
            decoder_vocab_count,
            model_dim,
        )
        self.decoder_blocks = nn.ModuleList([])
        for _ in range(decoder_block_count):
            self.decoder_blocks.append(TransformerDecoderBlock(
                model_dim, head_count, feed_forward_hidden_dim
            ))

        self.output_layer = nn.Linear(model_dim, decoder_vocab_count)

        if gpu_id >= 0:
            self.device = torch.device('cuda:{}'.format(gpu_id))
        else:
            self.device = torch.device('cpu')
        self.to(self.device)

    def encode(self, encoder_inputs):
        h = self.encoder_embedding_layer(encoder_inputs)
        if encoder_inputs.shape[1] is not None:
            positional_encodes = self.positional_encoder(encoder_inputs.shape[1])
            h += torch.Tensor(positional_encodes).to(self.device)
        encoder_masks = get_masks(encoder_inputs)

        for block in self.encoder_blocks:
            h = block(h, encoder_masks)

        return h, encoder_masks

    def decode(self, decoder_inputs, encoder_outputs, encoder_masks):
        h = self.decoder_embedding_layer(decoder_inputs)
        if decoder_inputs.shape[1] is not None:
            positional_encodes = self.positional_encoder(decoder_inputs.shape[1])
            h += torch.Tensor(positional_encodes).to(self.device)
        decoder_masks = get_masks(decoder_inputs)

        for block in self.decoder_blocks:
            h = block(h, encoder_outputs, decoder_masks, encoder_masks)

        h = self.output_layer(h)

        return h

    def forward(self, encoder_inputs, decoder_inputs):
        encoder_outputs, encoder_masks = self.encode(encoder_inputs)
        decoder_outputs = self.decode(decoder_inputs, encoder_outputs, encoder_masks)

        return decoder_outputs

    def inference(self, encoder_inputs, decoding_params):
        state = None

        inference_decoder = get_inference_decoder(decoding_params, encoder_inputs.shape[1])
        seq_len = decoding_params['seq_len']

        encoder_outputs, _ = self.encode(encoder_inputs)
        encoder_masks = get_masks(encoder_inputs)

        input_candidates = inference_decoder.start(state)
        for seq in range(seq_len):
            output_candidates = []
            for _, _, paths, probs in input_candidates:
                decoder_inputs = torch.LongTensor(paths[:, :seq+1].transpose(1, 0)).to(self.device)
                all_outputs = self.decode(decoder_inputs, encoder_outputs, encoder_masks)
                output_candidates.append(((all_outputs[-1], state), all_outputs, paths, probs))

            input_candidates = inference_decoder.decode(output_candidates)

        return input_candidates[0][1], input_candidates[0][2][:, 1:]
