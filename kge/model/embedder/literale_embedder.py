import torch
from torch import Tensor

from kge import Config, Dataset
from kge.job import Job
from kge.model import KgeEmbedder

from typing import List

class LiteralEGate(torch.nn.Module):
    """
    Taken from https://github.com/SmartDataAnalytics/LiteralE/blob/master/model.py
    and rewritten to shorten.
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 gate_activation=torch.sigmoid):

        super().__init__()

        self.gate_activation = gate_activation
        self.z = torch.nn.Linear(input_dim, output_dim)
        self.h = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x_cat, x_struct):
        h = torch.tanh(self.h(x_cat))
        z = self.gate_activation(self.z(x_cat))
        output = z * h + (1-z) * x_struct

        return output

class LiteralEEmbedder(KgeEmbedder):
    """Adds the LiteralE gate to a base embedder."""

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
        self.regularize = self.check_option("regularize", ["", "lp"])
        self.config.check("train.trace_level", ["batch", "epoch"])
        self.vocab_size = vocab_size

        self.gate_input_dim = 0

        if "struct" not in self.get_option("modalities"):
            raise ValueError("literale_embedder needs struct as modality")
        
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
            self.gate_input_dim += embedder.dim
            self.embedder[modality] = embedder

        # HACK
        # kwargs["indexes"] is set to None, if literale_embedder has
        # regularize_args.weighted set to False.
        # If the child_embedder has regularize_args.weighted set to True,
        # it tries to access kwargs["indexes"], which leads to an error

        # Set regularize_args.weighted to True, if it is set for the struct embedder
        if self.embedder["struct"].get_option("regularize_args.weighted"):
            config.set(
                self.configuration_key + ".regularize_args.weighted",
                True
            )

        # initialize gate
        self.gate = LiteralEGate(
            self.gate_input_dim, self.dim
        )
        
        if not init_for_load_only:
            # initialize weights
            for name, weights in self.gate.named_parameters():
                # set bias to zero
                # https://cs231n.github.io/neural-networks-2/#init 
                if "bias" in name:
                    torch.nn.init.zeros_(weights)
                else:
                    self.initialize(weights)

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
            embedding = self.embedder[modality].embed(indexes)
            embeddings_list.append(embedding)
            if modality == "struct":
                embedding_struct = embedding
        embeddings = torch.cat(embeddings_list, dim=1)
        
        return self.gate(embeddings, embedding_struct)

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

    def _get_regularize_weight(self) -> Tensor:
        return self.get_option("regularize_weight")

    def penalty(self, **kwargs) -> List[Tensor]:
        result = super().penalty(**kwargs)
        # get penalty of modality embedders
        for modality in self.get_option("modalities"):
            result += self.embedder[modality].penalty(**kwargs)
        
        # get penalty of gate
        if self.regularize == "" or self.get_option("regularize_weight") == 0.0:
            pass
        elif self.regularize == "lp":
            p = (
                self.get_option("regularize_args.p")
                if self.has_option("regularize_args.p")
                else 2
            )
            regularize_weight = self._get_regularize_weight()
            # unweighted Lp regularization
            parameter_list = []
            for name, parameter in self.gate.named_parameters():
                if "weight" in name:
                    parameter_list.append(torch.flatten(parameter))
            parameters = torch.cat(parameter_list)

            result += [
                (
                    f"{self.configuration_key}.L{p}_penalty",
                    (regularize_weight / p * parameters.norm(p=p) ** p).sum(),
                )
            ]
        else:  # unknown regularization
            raise ValueError(f"Invalid value regularize={self.regularize}")

        return result