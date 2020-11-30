import torch
import torch.nn
from torch import Tensor
import math

from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel


class TransformerScorer(RelationalScorer):
    r"""Implementation of the plain Transformer encode scorer.
    Must be used with ReciprocalRelationsModel."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)
        self.emb_dim = self.get_option("entity_embedder.dim")

        # the CLS embedding
        self.cls_emb = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        self.sub_type_emb = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        self.rel_type_emb = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))

        # TODO make all parameters configurable
        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.emb_dim, nhead=8, dim_feedforward=1280, dropout=0.1
            )
        self.encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=3)
        for layer in self.encoder.layers:
            self.initialize(layer.linear1.weight.data)
            self.initialize(layer.linear2.weight.data)

            if layer.self_attn._qkv_same_embed_dim:
                self.initialize(layer.self_attn.in_proj_weight)
            else:
                self.initialize(layer.self_attn.q_proj_weight)
                self.initialize(layer.self_attn.k_proj_weight)
                self.initialize(layer.self_attn.v_proj_weight)

    def score_emb(self, s_emb, p_emb, o_emb, combine: str):
        if combine not in ["sp_", "spo"]:
            raise ValueError(
                "Combine {} not supported in Transformer's score function".format(
                    combine
                )
            )

        # transform the sp pairs
        batch_size = len(s_emb)
        out = self.encoder.forward(
            torch.stack(
                (
                    self.cls_emb.repeat((batch_size, 1)),
                    s_emb + self.sub_type_emb.unsqueeze(0),
                    p_emb + self.rel_type_emb.unsqueeze(0),
                ),
                dim=0,
            )
        )  # SxNxE = 3 x batch_size x emb_size

        # pick the transformed CLS embeddings
        out = out[0, ::]

        # now take dot product
        if combine == "sp_":
            out = torch.mm(out, o_emb.transpose(1, 0))
        elif combine == "spo":
            out = (out * o_emb).sum(-1)
        else:
            raise Exception("can't happen")

        # all done
        return out.view(batch_size, -1)


class Transformer(KgeModel):
    r"""Implementation of the Transformer KGE model."""

    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key=None,
        init_for_load_only=False,
    ):
        self._init_configuration(config, configuration_key)
        super().__init__(
            config=config,
            dataset=dataset,
            scorer=TransformerScorer(config, dataset, self.configuration_key),
            configuration_key=self.configuration_key,
            init_for_load_only=init_for_load_only,
        )

    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None) -> Tensor:
        # We overwrite this method to ensure that ConvE only predicts towards objects.
        # If Transformer is wrapped in a reciprocal relations model, this will always be
        # the case.
        if direction == "o":
            super().score_spo(s, p, o, direction)
        else:
            raise ValueError("Transformer can only score objects")