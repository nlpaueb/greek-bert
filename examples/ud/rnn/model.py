import pytorch_wrapper as pw

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class UDRNNModel(nn.Module):

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
        super(UDRNNModel, self).__init__()
        self._embedding_layer = pw.modules.EmbeddingLayer(embeddings.shape[0], embeddings.shape[1], False, 0)
        self._embedding_layer.load_embeddings(embeddings)

        self._rnn = rnn_class(
            input_size=embeddings.shape[1],
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

    def forward(self, texts, text_lens):
        texts = self._embedding_layer(texts)
        texts = pack_padded_sequence(texts, text_lens, batch_first=True, enforce_sorted=False)
        texts = self._rnn(texts)[0]
        texts = pad_packed_sequence(texts, batch_first=True)[0]
        texts = self._rnn_top_layer_dp(texts)
        return self._out_mlp(texts)
