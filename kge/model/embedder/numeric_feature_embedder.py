from torch import Tensor
import torch

from kge import Config, Dataset
from kge.job import Job
from kge.model import KgeEmbedder

from typing import List

class NumericMLP(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
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
        self.lin.append(torch.nn.Linear(input_dim,output_dim))
        for i in range(1,num_layers):
            self.lin.append(torch.nn.Linear(output_dim,output_dim))

    def forward(self, numeric_data):
        output = numeric_data
        for l in self.lin:
            output = l(output)
            output = self.activation(output)

        return output

class NumericFeatureEmbedder(KgeEmbedder):
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
        self.filename = self.get_option("filename")
        self.num_layers = self.get_option("num_layers")

        # load numeric data
        with open(self.filename,"r") as f:
            data = list(
                map(lambda s: s.strip().split("\t"), f.readlines())
            )
        
        # returns entities in index order
        entities = self.dataset.entity_ids()
        
        ent_to_idx ={ent: idx for idx, ent in enumerate(entities)}
        numeric_data_ent_idx = []

        rel_to_idx = {}
        numeric_data_rel_idx = []
        numeric_data = []
        for t in data:
            ent = t[0]
            rel = t[1]
            value = float(t[2])

            if rel not in rel_to_idx:
                rel_to_idx[rel] = len(rel_to_idx)
            
            numeric_data_ent_idx.append(ent_to_idx[ent])
            numeric_data_rel_idx.append(rel_to_idx[rel])
            numeric_data.append(value)

        numeric_data_ent_idx = torch.tensor(
            numeric_data_ent_idx, dtype=torch.long
        ) 
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

                # account for the fact that there might only be a single value
                # in that case torch.std would result in nan
                if torch.sum(sel) > 1:
                    std = torch.std(numeric_data[sel]) 
                else:
                    std = 0
                
                numeric_data[sel] = (
                    (numeric_data[sel] - mean) / (std + 1e-8)
                )
        else:
            raise ValueError("Unkown normalization option")

        num_lit = torch.zeros(
            [len(ent_to_idx), len(rel_to_idx)], dtype=torch.float32
        )

        num_lit[numeric_data_ent_idx,numeric_data_rel_idx] = numeric_data
        # includes all numeric literals for all entities, with the entities
        # being ordered by their index
        self.num_lit = num_lit.to(self.config.get("job.device"))

        if self.num_layers > 0:
            # initialize numeric MLP
            self.numeric_mlp = NumericMLP(
                input_dim=num_lit.shape[1],
                output_dim=self.dim,
                num_layers=self.num_layers,
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
        else:
            self.dim = num_lit.shape[1]

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
        numeric_data = self.num_lit[indexes,:]
        if self.num_layers > 0:
            numeric_data = self.numeric_mlp(numeric_data)
        return numeric_data
    
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
        if (
            self.regularize == "" or
            self.get_option("regularize_weight") == 0.0 or
            self.num_layers <= 0
        ):
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