import torch
import torch.nn as nn
from torch import Tensor

from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel


class TransformerScorer(RelationalScorer):
    r"""Implementation of the plain Transformer encode scorer.
    Must be used with ReciprocalRelationsModel."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)

        from transformers.models.bert.configuration_bert import BertConfig
        self.bert_config = BertConfig()

        #self.bert_config.hidden_size = 320
        self.bert_config.hidden_size = 256
        #self.bert_config.hidden_size = 200

        self.bert_config.num_attention_heads = 8
        self.bert_config.num_hidden_layers = 3
        self.bert_config.intermediate_size = 1280

        self.bert_config.attention_probs_dropout_prob = .1
        self.bert_config.hidden_dropout_prob = .1
        self.bert_config.embedding_dropout_prob = .6
        #self.bert_config.embedding_dropout_prob = .2

        del self.bert_config.max_position_embeddings
        del self.bert_config.pad_token_id
        del self.bert_config.type_vocab_size
        del self.bert_config.vocab_size

        self.special_embeddings = nn.Embedding(2, self.bert_config.hidden_size) #CLS, MASK
        self.token_type_embeddings = nn.Embedding(3, self.bert_config.hidden_size)
        self.LayerNorm = nn.LayerNorm(self.bert_config.hidden_size, eps=self.bert_config.layer_norm_eps)
        self.dropout = nn.Dropout(self.bert_config.embedding_dropout_prob) 

        self.emb_dim = self.get_option("entity_embedder.dim")

        # the CLS embedding
        self.cls_emb = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        # add CLS type embedding???
        self.sub_type_emb = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        self.rel_type_emb = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))

        # TODO make all parameters configurable
        from transformers.models.bert.modeling_bert import BertEncoder
        self.encoder = BertEncoder(self.bert_config)

        def _init_weights(module):
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=self.bert_config.initializer_range)
            elif isinstance(module, torch.nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        self.apply(_init_weights)

    def score_emb(self, s_emb, p_emb, o_emb, combine: str):
        if combine not in ["sp_", "spo"]:
            raise ValueError(
                "Combine {} not supported in Transformer's score function".format(
                    combine
                )
            )

        # transform the sp pairs
        batch_size = len(s_emb)
        device = self.special_embeddings.weight.device

        cls_embeds = self.special_embeddings(torch.tensor(0).to(device)).unsqueeze(0).expand(batch_size, -1)
        input_embeds = torch.cat([cls_embeds.unsqueeze(1),
                                  s_emb.unsqueeze(1),
                                  p_emb.unsqueeze(1)], 1)
        token_type_ids = torch.arange(3).unsqueeze(0).expand(batch_size, -1).to(device)
        token_type_embeds = self.token_type_embeddings(token_type_ids)

        embeds = input_embeds + token_type_embeds
        embeds = self.LayerNorm(embeds)
        embeds = self.dropout(embeds)

        encoder_output = self.encoder(embeds, head_mask=[None]*self.bert_config.num_hidden_layers)[0]
        cls_output = encoder_output[:, 0]

        # now take dot product
        if combine == "sp_":
            out = torch.mm(cls_output, o_emb.transpose(1, 0))
        elif combine == "spo":
            out = (cls_output * o_emb).sum(-1)
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