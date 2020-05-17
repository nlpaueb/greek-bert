import torch
import torch.nn.functional as F
import pytorch_wrapper as pw
import pytorch_wrapper.functional as pwF

from torch import nn


class XNLIDAMModel(nn.Module):

    def __init__(self,
                 embeddings,
                 mlp_num_layers=1,
                 mlp_hidden_size=200,
                 mlp_activation=nn.ReLU,
                 mlp_dp=0.2):
        super(XNLIDAMModel, self).__init__()
        self._embedding_layer = pw.modules.EmbeddingLayer(embeddings.shape[0], embeddings.shape[1], False, 0)
        self._embedding_layer.load_embeddings(embeddings)

        self._linear_projection = nn.Linear(embeddings.shape[1], mlp_hidden_size, bias=False)

        self._att_mlp = pw.modules.MLP(
            input_size=mlp_hidden_size,
            num_hidden_layers=mlp_num_layers,
            hidden_layer_size=mlp_hidden_size,
            hidden_activation=mlp_activation,
            hidden_dp=mlp_dp,
            output_size=mlp_hidden_size,
            output_activation=mlp_activation,
            hidden_layer_init=lambda x, y: torch.nn.init.xavier_uniform_(x),
            output_layer_init=lambda x, y: torch.nn.init.xavier_uniform_(x)
        )

        self._comp_mlp = pw.modules.MLP(
            input_size=2 * mlp_hidden_size,
            num_hidden_layers=mlp_num_layers,
            hidden_layer_size=mlp_hidden_size,
            hidden_activation=mlp_activation,
            hidden_dp=mlp_dp,
            output_size=mlp_hidden_size,
            output_activation=mlp_activation,
            hidden_layer_init=lambda x, y: torch.nn.init.xavier_uniform_(x),
            output_layer_init=lambda x, y: torch.nn.init.xavier_uniform_(x)
        )

        self._out_mlp = pw.modules.MLP(
            input_size=2 * mlp_hidden_size,
            num_hidden_layers=mlp_num_layers,
            hidden_layer_size=mlp_hidden_size,
            hidden_activation=mlp_activation,
            hidden_dp=mlp_dp,
            output_size=3,
            output_activation=None,
            hidden_layer_init=lambda x, y: torch.nn.init.xavier_uniform_(x),
            output_layer_init=lambda x, y: torch.nn.init.xavier_uniform_(x)
        )

    def forward(self, prems_indexes, prem_lens, hypos_indexes, hypo_lens):
        prems = self._embedding_layer(prems_indexes)
        prems = prems / (prems.norm(dim=-1, keepdim=True) + 1e-6)
        prems = self._linear_projection(prems)
        prems_mask = pwF.create_mask_from_length(prem_lens, prems.shape[1])
        prems_att_vectors = self._att_mlp(prems)

        hypos = self._embedding_layer(hypos_indexes)
        hypos = hypos / (hypos.norm(dim=-1, keepdim=True) + 1e-6)
        hypos = self._linear_projection(hypos)
        hypos_mask = pwF.create_mask_from_length(hypo_lens, hypos.shape[1])
        hypos_att_vectors = self._att_mlp(hypos)

        scores = torch.matmul(prems_att_vectors, hypos_att_vectors.transpose(1, 2))
        scores = scores.masked_fill(prems_mask.unsqueeze(2) == 0, -1e9)
        scores = scores.masked_fill(hypos_mask.unsqueeze(1) == 0, -1e9)

        horizontal_softmaxed = F.softmax(scores, dim=2)
        vertical_softmaxed = F.softmax(scores, dim=1)

        hypos_attended = torch.matmul(horizontal_softmaxed, hypos)
        prems_hypos_attended = torch.cat([prems, hypos_attended], dim=-1)
        prems_hypos_attended_compared = self._comp_mlp(prems_hypos_attended)
        prems_hypos_attended_compared = prems_hypos_attended_compared.masked_fill(prems_mask.unsqueeze(2) == 0, 0)
        prems_hypos_attended_aggregated = torch.sum(prems_hypos_attended_compared, dim=1)

        prems_attended = torch.matmul(vertical_softmaxed.transpose(1, 2), prems)
        hypos_prems_attended = torch.cat([hypos, prems_attended], dim=-1)
        hypos_prems_attended_compared = self._comp_mlp(hypos_prems_attended)
        hypos_prems_attended_compared = hypos_prems_attended_compared.masked_fill(hypos_mask.unsqueeze(2) == 0, 0)
        hypos_prems_attended_aggregated = torch.sum(hypos_prems_attended_compared, dim=1)

        encodings = torch.cat([prems_hypos_attended_aggregated, hypos_prems_attended_aggregated], dim=-1)

        return self._out_mlp(encodings)
