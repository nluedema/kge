from torch import Tensor
import torch.nn
import torch.nn.functional

from kge import Config, Dataset
from kge.job import Job
from kge.model import KgeEmbedder
from kge.misc import round_to_points

from typing import List

class NumericMLP(torch.nn.Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()

        self.lin_1 = torch.nn.Linear(1,dim)
        self.lin_2 = torch.nn.Linear(dim,dim)

    def forward(self, numeric_data):
        lin_1 = torch.tanh(self.lin_1(numeric_data))
        lin_2 = torch.tanh(self.lin_2(lin_1))

        return lin_2


class NumericEmbedder(KgeEmbedder):
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

        if self.dim < 0:
            raise ValueError ("dimension was not initialized by multimodal_embedder")
        
        self.start_idx = self.get_option("start_idx")
        if self.dim < 0:
            raise ValueError ("start_idx was not initialized by multimodal_embedder")

        self.filename = self.get_option("filename")
        if self.filename == "":
            raise ValueError ("filename was not initialized by multimodal_embedder")

        # load numeric triples
        with open(self.filename,"r") as f:
            data = list(
                map(lambda s: s.strip().split("\t"), f.readlines())
            )
        
        numeric_data = []
        for t in data:
            numeric_data.append(float(t[2]))
        numeric_data = torch.tensor(
            numeric_data,dtype=torch.float32,device=config.get("job.device")
        )

        # normalization
        normalization = self.get_option("normalization")
        if normalization == "z-score":
            mean = numeric_data.mean()
            std = numeric_data.std()
            numeric_data = (numeric_data - mean) / std
        
        self.numeric_data = numeric_data

        # initialize numeric MLP
        self.numeric_mlp = NumericMLP(self.dim)

        if not init_for_load_only:
            # initialize weights
            for name, weights in self.numeric_mlp.named_parameters():
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

    def _embed(self, indexes: Tensor) -> Tensor:
            indexes = indexes - self.start_idx
            
            numeric_data = self.numeric_data[indexes]
            numeric_data = numeric_data.reshape(-1,1)
            return self.numeric_mlp(numeric_data)
    
    def embed(self, indexes: Tensor) -> Tensor:
        return self._postprocess(self._embed(indexes.long()))

    def embed_all(self) -> Tensor:
        raise ValueError("Should never be called, in multimodal setting")

    def _postprocess(self, embeddings: Tensor) -> Tensor:
        if self.dropout.p > 0:
            embeddings = self.dropout(embeddings)
        return embeddings

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
            for name, parameter in self.numeric_mlp.named_parameters():
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