import pytorch_wrapper as pw
import pytorch_wrapper.functional as pwF
import torch

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchcrf import CRF

from ...utils.sequences import pad_to_max


class UDRNNModel(nn.Module):

    def __init__(self,
                 char_embeddings_shape,
                 embeddings,
                 char_cnn_kernel_heights=(3,),
                 char_cnn_out_channels=30,
                 rnn_class=nn.GRU,
                 rnn_hidden_size=128,
                 rnn_num_layers=2,
                 rnn_dp=0.2,
                 rnn_bidirectional=True,
                 mlp_num_layers=1,
                 mlp_hidden_size=128,
                 mlp_activation=nn.ReLU,
                 mlp_dp=0.2):
        super(UDRNNModel, self).__init__()

        self._char_embedding_layer = pw.modules.EmbeddingLayer(
            char_embeddings_shape[0],
            char_embeddings_shape[1],
            True,
            0
        )

        self._char_token_encoder = pw.modules.sequence_basic_cnn_encoder.SequenceBasicCNNEncoder(
            char_embeddings_shape[1],
            kernel_heights=char_cnn_kernel_heights,
            out_channels=char_cnn_out_channels
        )

        self._embedding_layer = pw.modules.EmbeddingLayer(embeddings.shape[0], embeddings.shape[1], False, 0)
        self._embedding_layer.load_embeddings(embeddings)

        self._embedding_dp = nn.Dropout(rnn_dp)

        self._rnn = rnn_class(
            input_size=embeddings.shape[1] + char_cnn_out_channels * len(char_cnn_kernel_heights),
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            dropout=rnn_dp,
            bidirectional=rnn_bidirectional,
            batch_first=True
        )

        self._rnn_top_layer_dp = nn.Dropout(rnn_dp)

        self._out_mlp = pw.modules.MLP(
            input_size=rnn_hidden_size * (2 if rnn_bidirectional else 1),
            num_hidden_layers=mlp_num_layers,
            hidden_layer_size=mlp_hidden_size,
            hidden_activation=mlp_activation,
            hidden_dp=mlp_dp,
            output_size=17,
            output_activation=None
        )

        self._crf = CRF(17, batch_first=True)

    def forward(self,
                batched_char_words,
                batched_char_words_len,
                batched_char_word_index,
                batched_tokens,
                batched_tokens_len,
                target=None
                ):

        char_tokens = self._char_embedding_layer(batched_char_words)
        token_encodings = self._char_token_encoder(char_tokens)

        token_encodings_z = torch.zeros((1, token_encodings.shape[1]), device=token_encodings.device)
        token_encodings = torch.cat([token_encodings_z, token_encodings], dim=0)

        token_encodings_indexed = torch.index_select(token_encodings, dim=0, index=batched_char_word_index.view(-1))
        token_encodings_indexed = token_encodings_indexed.view(
            batched_char_word_index.shape[0],
            batched_char_word_index.shape[1],
            -1
        )

        texts = self._embedding_layer(batched_tokens)
        texts = torch.cat([texts, token_encodings_indexed], -1)
        texts = self._embedding_dp(texts)

        texts = pack_padded_sequence(texts, batched_tokens_len, batch_first=True, enforce_sorted=False)
        texts = self._rnn(texts)[0]
        texts = pad_packed_sequence(texts, batch_first=True)[0]
        texts = self._rnn_top_layer_dp(texts)
        mlp_out = self._out_mlp(texts)

        mask = pwF.create_mask_from_length(batched_tokens_len, mlp_out.shape[1])

        if self.training:
            return -self._crf(mlp_out, target, mask=mask, reduction='token_mean')
        else:
            predictions = self._crf.decode(mlp_out, mask)
            predictions = torch.tensor(pad_to_max(predictions), dtype=torch.long).to(mlp_out.device)
            one_hot_pred = torch.eye(17).to(mlp_out.device)[[predictions]]
            return one_hot_pred
