from torch import Tensor
import torch.nn
import torch.nn.functional

from kge import Config, Dataset
from kge.job import Job
from kge.model import KgeEmbedder
from kge.misc import round_to_points

from typing import List

import numpy as np

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

    def forward(self, x_ent, x_lit):
        x = torch.cat([x_ent, x_lit], 1)
        h = torch.tanh(self.h(x))
        z = self.gate_activation(self.z(x))
        output = z * h + (1-z) * x_ent

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

        # initialize base_embedder
        if self.configuration_key + ".base_embedder.type" not in config.options:
            config.set(
                self.configuration_key + ".base_embedder.type",
                self.get_option("base_embedder.type"),
            )
        self.base_embedder = KgeEmbedder.create(
            config, dataset, self.configuration_key + ".base_embedder", self.vocab_size
        )
        if self.dim < 0:
            # set dim to base_embedder dim
            self.dim = self.base_embedder.dim

        # load numeric literals
        import os.path
        local_path = "/home/niklas/Desktop/kge/LiteralE_additional_files/fb15k-237/numerical_literals.npy"
        gpu_path = "/home/nluedema/kge/LiteralE_additional_files/fb15k-237/numerical_literals.npy"
        
        if os.path.isfile(gpu_path):
            self.num_lit = np.load(gpu_path)
        elif os.path.isfile(local_path):
            self.num_lit = np.load(local_path)
        else:
            raise FileNotFoundError("numeric literal file cannot be found")

        self.dim_lit = self.num_lit.shape[1]
        
        # normalize numeric literals
        max_lit = np.max(self.num_lit, axis=0)
        min_lit = np.min(self.num_lit, axis=0)
        self.num_lit = (self.num_lit - min_lit) / (max_lit - min_lit + 1e-8)

        # transform to tensor
        self.num_lit = torch.nn.Parameter(torch.from_numpy(self.num_lit))

        # initialize gate
        self.gate = LiteralEGate(self.base_embedder.dim + self.dim_lit, self.dim)

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
        self.base_embedder.prepare_job(job, **kwargs)
    
    def _embed(self, indexes: Tensor) -> Tensor:
        embeddings = self.base_embedder.embed(indexes)
        num_lit = self.num_lit[indexes,:]

        return  self.gate(embeddings, num_lit)

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
                self.vocab_size, dtype=torch.long, device=self.base_embedder._embeddings.weight.device
            )
        )

    def _get_regularize_weight(self) -> Tensor:
        return self.get_option("regularize_weight")

    def penalty(self, **kwargs) -> List[Tensor]:
        # TODO factor out to a utility method
        result = super().penalty(**kwargs)
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

        return result + self.base_embedder.penalty(**kwargs)
