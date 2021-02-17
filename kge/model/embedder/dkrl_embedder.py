import torch
from torch import Tensor

from kge import Config, Dataset
from kge.job import Job
from kge.model import KgeEmbedder

from typing import List

class DKRLEmbedder(KgeEmbedder):

    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key: str,
        vocab_size: int,
        init_for_load_only=False,
    ):
        super().__init__(
            config, dataset, configuration_key, init_for_load_only=init_for_load_only
        )

        # read config
        self.config.check("train.trace_level", ["batch", "epoch"])
        self.vocab_size = vocab_size

        if self.get_option("modalities")[0] != "struct":
            raise ValueError("DKRL assumes that struct is the first modality")

        # set relation embedder dim
        # fixes the problem that for the search, relation and entity embeder dim
        # has to be set with a single config
        # CAREFULL: THIS ASSUMES THAT THE ENITY EMBEDER IS CREATED FIRST
        rel_emb_conf_key = configuration_key.replace(
            "entity_embedder", "relation_embedder"
        )
        if configuration_key == rel_emb_conf_key:
            raise ValueError(
                "Cannot set the relation embedding size"
            )
        config.set(
            f"{rel_emb_conf_key}.dim", self.dim
        )

        # create embedder for each modality
        self.embedder = torch.nn.ModuleDict()
        for modality in self.get_option("modalities"):
            # if dim of modality embedder is < 0 set it to parent embedder dim
            # e.g. when using dkrl, the text embedding dim should equal embedding dim
            # but when using literale, the text embedding dim can vary
            if self.get_option(f"{modality}.dim") < 0:
                config.set(
                    f"{self.configuration_key}.{modality}.dim",
                    self.dim
                )
            
            embedder = KgeEmbedder.create(
                config, dataset, f"{self.configuration_key}.{modality}",
                vocab_size=self.vocab_size, init_for_load_only=init_for_load_only
            )
            self.embedder[modality] = embedder
        
        # HACK
        # kwargs["indexes"] is set to None, if dkrl_embedder has
        # regularize_args.weighted set to False.
        # If the child_embedder has regularize_args.weighted set to True,
        # it tries to access kwargs["indexes"], which leads to an error

        # Set regularize_args.weighted to True, if it is set for the struct embedder
        if self.embedder["struct"].get_option("regularize_args.weighted"):
            config.set(
                self.configuration_key + ".regularize_args.weighted",
                True
            )
        
        # TODO handling negative dropout because using it with ax searches for now
        dropout = self.get_option("dropout")
        if dropout < 0:
            if config.get("train.auto_correct"):
                config.log(
                    "Setting {}.dropout to 0, "
                    "was set to {}.".format(configuration_key, dropout)
                )
                dropout = 0
        self.dropout = torch.nn.Dropout(dropout)

    def prepare_job(self, job: Job, **kwargs):
        super().prepare_job(job, **kwargs)
        for modality in self.get_option("modalities"):
            self.embedder[modality].prepare_job(job, **kwargs)
    
    def _embed(self, indexes: Tensor) -> Tensor:
        embeddings_list = []
        for modality in self.get_option("modalities"):
            embeddings_list.append(self.embedder[modality].embed(indexes))
        # concatenate the embeddings along a new dimension
        # creates tensor of shape N*dim*(nbr of modalities) 
        embeddings = torch.stack(embeddings_list, dim=2)

        return embeddings
    
    def embed(self, indexes: Tensor) -> Tensor:
        return self._postprocess(self._embed(indexes.long()))

    def embed_all(self) -> Tensor:
        return self._postprocess(self._embeddings_all())

    def _postprocess(self, embeddings: Tensor) -> Tensor:
        if self.dropout.p > 0:
            embeddings = self.dropout(embeddings)
        return embeddings

    def _embeddings_all(self) -> Tensor:
        return self._embed(
            torch.arange(
                self.vocab_size, dtype=torch.long, device=self.config.get("job.device")
            )
        )

    def penalty(self, **kwargs) -> List[Tensor]:
        result = super().penalty(**kwargs)
        # get penalty of modality embedders
        for modality in self.get_option("modalities"):
            result += self.embedder[modality].penalty(**kwargs)
        
        return result