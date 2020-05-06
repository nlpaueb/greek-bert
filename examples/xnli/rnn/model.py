import torch
import pytorch_wrapper as pw
import pytorch_wrapper.functional as pwF

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class XNLIRNNModel(nn.Module):

    def __init__(self,
                 embeddings,
                 rnn_class=nn.GRU,
                 rnn_hidden_size=128,
                 rnn_num_layers=2,
                 rnn_dp=0.2,
                 rnn_bidirectional=True,
                 mlp_num_layers=1,
                 mlp_hidden_size=128,
                 mlp_activation=nn.ReLU,
                 mlp_dp=0.2):
        super(XNLIRNNModel, self).__init__()
        self._embedding_layer = pw.modules.EmbeddingLayer(embeddings.shape[0], embeddings.shape[1], False, 0)
        self._embedding_layer.load_embeddings(embeddings)

        self._prem_rnn = rnn_class(
            input_size=embeddings.shape[1],
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            dropout=rnn_dp,
            bidirectional=rnn_bidirectional,
            batch_first=True
        )

        self._hypo_rnn = rnn_class(
            input_size=embeddings.shape[1],
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            bidirectional=rnn_bidirectional,
            dropout=rnn_dp,
            batch_first=True
        )

        self._rnn_top_layer_dp = nn.Dropout(rnn_dp)

        self._out_mlp = pw.modules.MLP(
            input_size=2 * rnn_hidden_size * (2 if rnn_bidirectional else 1),
            num_hidden_layers=mlp_num_layers,
            hidden_layer_size=mlp_hidden_size,
            hidden_activation=mlp_activation,
            hidden_dp=mlp_dp,
            output_size=3,
            output_activation=None
        )

    def forward(self, prems, prem_lens, hypos, hypo_lens):
        prems = self._embedding_layer(prems)
        prems = pack_padded_sequence(prems, prem_lens, batch_first=True, enforce_sorted=False)
        prems = self._prem_rnn(prems)[0]
        prems = pad_packed_sequence(prems, batch_first=True)[0]
        prems_mask = pwF.create_mask_from_length(prem_lens, prems.shape[1])
        prem_encodings = pwF.masked_mean_pooling(prems, prems_mask, dim=1)

        hypos = self._embedding_layer(hypos)
        hypos = pack_padded_sequence(hypos, hypo_lens, batch_first=True, enforce_sorted=False)
        hypos = self._hypo_rnn(hypos)[0]
        hypos = pad_packed_sequence(hypos, batch_first=True)[0]
        hypos_mask = pwF.create_mask_from_length(hypo_lens, hypos.shape[1])
        hypo_encodings = pwF.masked_mean_pooling(hypos, hypos_mask, dim=1)

        encodings = torch.cat([prem_encodings, hypo_encodings], dim=-1)

        return self._out_mlp(encodings)
