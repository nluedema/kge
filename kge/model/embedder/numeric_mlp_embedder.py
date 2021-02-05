from torch import Tensor
import torch

from kge import Config, Dataset
from kge.job import Job
from kge.model import KgeEmbedder

from typing import List

class NumericMLP(torch.nn.Module):
    def __init__(
        self,
        dim,
        num_layers,
        activation,
    ):
        super().__init__()
        if activation == "tanh":
            self.activation = torch.nn.Tanh()
        elif activation == "relu":
            self.activation = torch.nn.ReLU()
        else:
            raise ValueError("unkown activation function specified")

        self.lin = torch.nn.ModuleList([])
        self.lin.append(torch.nn.Linear(1,dim))
        for i in range(1,num_layers):
            self.lin.append(torch.nn.Linear(dim,dim))

    def forward(self, numeric_data):
        output = numeric_data
        for l in self.lin:
            output = l(output)
            output = self.activation(output)

        return output


class NumericMLPEmbedder(KgeEmbedder):
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
            raise ValueError ("dim was not initialized by multimodal_embedder")
        
        self.start_idx = self.get_option("start_idx")
        if self.start_idx < 0:
            raise ValueError ("start_idx was not initialized by multimodal_embedder")

        self.filename = self.get_option("filename")
        if self.filename == "":
            raise ValueError ("filename was not initialized by multimodal_embedder")

        # load numeric data
        with open(self.filename,"r") as f:
            data = list(
                map(lambda s: s.strip().split("\t"), f.readlines())
            )
        
        rel_to_idx = {}
        numeric_data_rel_idx = []
        numeric_data = []
        for t in data:
            rel = t[1]
            value = float(t[2])

            if rel not in rel_to_idx:
                rel_to_idx[rel] = len(rel_to_idx)
            
            numeric_data_rel_idx.append(rel_to_idx[rel])
            numeric_data.append(value)

        numeric_data_rel_idx = torch.tensor(
            numeric_data_rel_idx, dtype=torch.long
        ) 
        numeric_data = torch.tensor(
            numeric_data, dtype=torch.float32
        )

        # normalize numeric literals
        if self.get_option("normalization") == "min-max":
            for rel_idx in rel_to_idx.values():
                sel = (rel_idx == numeric_data_rel_idx)
                max_num = torch.max(numeric_data[sel]) 
                min_num = torch.min(numeric_data[sel]) 
                numeric_data[sel] = (
                    (numeric_data[sel] - min_num) / (max_num - min_num + 1e-8)
                )
        elif self.get_option("normalization") == "z-score":
            for rel_idx in rel_to_idx.values():
                sel = (rel_idx == numeric_data_rel_idx)
                mean = torch.mean(numeric_data[sel]) 
                std = torch.std(numeric_data[sel]) 
                numeric_data[sel] = (
                    (numeric_data[sel] - mean) / std
                )
        else:
            raise ValueError("Unkown normalization option")

        self.numeric_data = numeric_data.to(self.config.get("job.device"))

        # initialize numeric MLP
        self.numeric_mlp = NumericMLP(
            dim=self.dim,
            num_layers=self.get_option("num_layers"),
            activation=self.get_option("activation")
        )

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