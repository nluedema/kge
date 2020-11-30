import torch
import torch.nn
from torch import Tensor

from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel


class TransformerScorer(RelationalScorer):
    r"""Implementation of the plain Transformer encode scorer.
    Must be used with ReciprocalRelationsModel."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)

        from transformers.models.bert.configuration_bert import BertConfig
        bert_config = BertConfig()

        bert_config.hidden_size = 320
        bert_config.num_attention_heads = 8
        bert_config.num_hidden_layers = 3
        bert_config.intermediate_size = 1280

        bert_config.attention_probs_dropout_prob = .1
        bert_config.hidden_dropout_prob = .1

        del bert_config.max_position_embeddings
        del bert_config.pad_token_id
        del bert_config.type_vocab_size
        del bert_config.vocab_size

        self.emb_dim = self.get_option("entity_embedder.dim")

        # the CLS embedding
        self.cls_emb = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        # add CLS type embedding???
        self.sub_type_emb = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        self.rel_type_emb = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))

        # TODO make all parameters configurable
        from transformers.models.bert.modeling_bert import BertEncoder
        self.encoder = BertEncoder(bert_config)

        def _init_weights(module):
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=bert_config.initializer_range)
            elif isinstance(module, torch.nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        self.encoder.apply(_init_weights)

    def score_emb(self, s_emb, p_emb, o_emb, combine: str):
        if combine not in ["sp_", "spo"]:
            raise ValueError(
                "Combine {} not supported in Transformer's score function".format(
                    combine
                )
            )

        # transform the sp pairs
        batch_size = len(s_emb)
        out = self.encoder(
            torch.stack(
                (
                    self.cls_emb.repeat((batch_size, 1)),
                    s_emb + self.sub_type_emb.unsqueeze(0),
                    p_emb + self.rel_type_emb.unsqueeze(0),
                ),
                dim=1,
            ),
            head_mask=[None] * 3
        )[0]  # NxSxE = batch_size x 3 x emb_size

        # pick the transformed CLS embeddings
        out = out[:, 0]

        # now take dot product
        if combine == "sp_":
            out = torch.mm(out, o_emb.transpose(1, 0))
        elif combine == "spo":
            out = (out * o_emb).sum(-1)
        else:
            raise Exception("can't happen")

        # all done
        return out.view(batch_size, -1)


class TransformerDavid(KgeModel):
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