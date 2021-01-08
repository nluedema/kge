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
        torch.nn.init.normal_(self.cls_emb, mean=0.0, std=0.02)
        self.type_emb_0 = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        torch.nn.init.normal_(self.type_emb_0, mean=0.0, std=0.02)
        self.type_emb_1 = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        torch.nn.init.normal_(self.type_emb_1, mean=0.0, std=0.02)
        self.type_emb_2 = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        torch.nn.init.normal_(self.type_emb_2, mean=0.0, std=0.02)

        # TODO make all parameters configurable
        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.emb_dim, nhead=8, dim_feedforward=1280, dropout=0.1,
            activation="gelu"
        )
        self.encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=3)
        for layer in self.encoder.layers:
            torch.nn.init.normal_(layer.linear1.weight.data, mean=0.0, std=0.02)
            torch.nn.init.zeros_(layer.linear1.bias.data)

            torch.nn.init.normal_(layer.linear2.weight.data, mean=0.0, std=0.02)
            torch.nn.init.zeros_(layer.linear2.bias.data)

            torch.nn.init.normal_(
                layer.self_attn.out_proj.weight.data, mean=0.0, std=0.02
            )
            torch.nn.init.zeros_(layer.self_attn.out_proj.bias.data)

            if layer.self_attn._qkv_same_embed_dim:
                torch.nn.init.normal_(
                    layer.self_attn.in_proj_weight, mean=0.0, std=0.02
                )
            else:
                torch.nn.init.normal_(
                    layer.self_attn.q_proj_weight, mean=0.0, std=0.02
                )
                torch.nn.init.normal_(
                    layer.self_attn.k_proj_weight, mean=0.0, std=0.02
                )
                torch.nn.init.normal_(
                    layer.self_attn.v_proj_weight, mean=0.0, std=0.02
                )
            
            if layer.self_attn.in_proj_bias is not None:
                torch.nn.init.zeros_(layer.self_attn.in_proj_bias)
            if layer.self_attn.bias_k is not None:
                torch.nn.init.zeros_(layer.self_attn.bias_k)
            if layer.self_attn.bias_v is not None:
                torch.nn.init.zeros_(layer.self_attn.bias_v)
        
        self.layer_norm = torch.nn.LayerNorm(self.emb_dim, eps=1e-12)
        self.layer_norm.bias.data.zero_()
        self.layer_norm.weight.data.fill_(1.0)

        self.dropout = torch.nn.Dropout(0.6)

    def score_emb(self, s_emb, p_emb, o_emb, combine: str):
        if combine not in ["sp_", "spo"]:
            raise ValueError(
                "Combine {} not supported in Transformer's score function".format(
                    combine
                )
            )

        # transform the sp pairs
        batch_size = len(s_emb)
        embeddings = torch.stack(
            (
                self.cls_emb.repeat((batch_size, 1)) + self.type_emb_0.unsqueeze(0),
                s_emb + self.type_emb_1.unsqueeze(0),
                p_emb + self.type_emb_2.unsqueeze(0),
            ),
            dim=0,
        ) # SxNxE = 3 x batch_size x emb_size
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        out = self.encoder.forward(embeddings)

        # pick the transformed CLS embeddings
        out = out[0, ::]

        o_emb = self.dropout(o_emb)

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
