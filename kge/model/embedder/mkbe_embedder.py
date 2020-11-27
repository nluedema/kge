from torch import Tensor
import torch.nn
import torch.nn.functional

from kge import Config, Dataset
from kge.job import Job
from kge.model import KgeEmbedder
from kge.misc import round_to_points

from typing import List

import numpy as np

class NumericalMLP(torch.nn.Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()

        self.lin_1 = torch.nn.Linear(1,dim)
        self.lin_2 = torch.nn.Linear(dim,dim)

    def forward(self, numerical_data):
        lin_1 = torch.tanh(self.lin_1(numerical_data))
        lin_2 = torch.tanh(self.lin_2(lin_1))

        return lin_2


class MKBEEmbedder(KgeEmbedder):
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

        # set base_embedder.dim to dim
        # do this because conve might round the dimension and then dim != base_embedder.dim
        # is there a better way???
        config.set(
            self.configuration_key + ".base_embedder.dim",
            self.dim
        )

        # initialize base_embedder
        if self.configuration_key + ".base_embedder.type" not in config.options:
            config.set(
                self.configuration_key + ".base_embedder.type",
                self.get_option("base_embedder.type"),
            )
        self.base_embedder = KgeEmbedder.create(
            config, dataset, self.configuration_key + ".base_embedder", self.vocab_size
        )

        # HACK
        # kwargs["indexes"] is set to None, if the mkbe_embedder has
        # regularize_args.weighted set to False.
        # If the base_embedder has regularize_args.weighted set to True,
        # it tries to access kwargs["indexes"], which leads to an error
        
        # Set regularize_args.weighted to True, if it is set for the base_embedder
        if self.base_embedder.get_option("regularize_args.weighted"):
            config.set(
                self.configuration_key + ".regularize_args.weighted",
                True
            )

        # load numerical literals
        import os.path
        local_path = "/home/niklas/Desktop/kge/mkbe_additional_files/yago3-10/numerical_data_preprocessed.npy"
        gpu_path = "/home/nluedema/kge/mkbe_additional_files/yago3-10/numerical_data_preprocessed.npy"

        if os.path.isfile(gpu_path):
            self.numerical_data = np.load(gpu_path)
        elif os.path.isfile(local_path):
            self.numerical_data = np.load(local_path)
        else:
            raise FileNotFoundError("numeric data file cannot be found")

        # transform to tensor
        self.numerical_data = torch.from_numpy(self.numerical_data).to(config.get("job.device"))

        # initialize numerical MLP
        self.numerical_mlp = NumericalMLP(self.dim)

        if not init_for_load_only:
            # initialize weights
            for name, weights in self.numerical_mlp.named_parameters():
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
        #embeddings = torch.empty(
        #    (len(indexes), self.dim), 
        #    device=self.base_embedder._embeddings.weight.device,
        #    requires_grad = True
        #)

        #indexes_structural_idx = indexes < 123182
        #indexes_numerical_idx = indexes >= 123182

        #indexes_structural = indexes[indexes_structural_idx]
        #indexes_numerical = indexes[indexes_numerical_idx] - 123182

        #embeddings_structural = self.base_embedder.embed(indexes_structural)
        #embeddings[indexes_structural_idx,:] = embeddings_structural

        #if len(indexes_numerical) > 0:
        #    numerical_data = self.numerical_data[indexes_numerical]

        #    # transform row into column vector
        #    numerical_data = numerical_data.reshape(-1,1)
        #    embeddings_numerical = self.numerical_mlp(numerical_data)
        #    embeddings[indexes_numerical_idx,:] = embeddings_numerical

        #return embeddings

        if torch.any(indexes < 123182).item():
            #if len(indexes) != sum(indexes < 123182).item():
            #    raise ValueError("all indices are expected to have the same modality")
            
            return self.base_embedder.embed(indexes)
        else:
            #if len(indexes) != sum(indexes >= 123182).item():
            #    raise ValueError("all indices are expected to have the same modality")
                    
            indexes_numerical = indexes - 123182
            
            numerical_data = self.numerical_data[indexes_numerical]
            numerical_data = numerical_data.reshape(-1,1)
            return self.numerical_mlp(numerical_data)
    
    def embed(self, indexes: Tensor) -> Tensor:
        return self._postprocess(self._embed(indexes.long()))

    def embed_all(self) -> Tensor:
        raise ValueError("Should never be called, because we distinguish between structural and numerical objects")
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
            for name, parameter in self.numerical_mlp.named_parameters():
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
