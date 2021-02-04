from torch import Tensor
import torch

from kge import Config, Dataset
from kge.job import Job
from kge.model import KgeEmbedder

from typing import List

import pandas

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
        self.config.check("train.trace_level", ["batch", "epoch"])
        self.vocab_size = vocab_size
        self.filename = self.get_option("filename")

        # process numeric data
        df = pandas.read_csv(self.filename, header=None, sep="\t")
    
        # returns entities in index order
        entities = self.dataset.entity_ids()

        ent_to_idx ={ent: idx for idx, ent in enumerate(entities)}
        rel_to_idx = {rel: idx for idx, rel in enumerate(df[1].unique())}

        num_lit = torch.zeros(
            [len(ent_to_idx), len(rel_to_idx)], dtype=torch.float32
        )

        for s, p, lit in df.values:
            try:
                num_lit[ent_to_idx[s.lower()], rel_to_idx[p]] = float(lit)
            except KeyError:
                print(f"Problem for:{s}\t{p}\t{lit}")
                continue
        
        # set dim
        self.dim = num_lit.shape[1]

        # normalize numeric literals
        if self.get_option("normalization") == "min-max":
            max_lit = torch.max(num_lit, dim=0)[0] # get values and ignore indices
            min_lit = torch.min(num_lit, dim=0)[0] # get values and ignore indices
            num_lit = (num_lit - min_lit) / (max_lit - min_lit + 1e-8)
        elif self.get_option("normalization") == "z-score":
            # do z-scores on each relation separately
            mean = torch.mean(num_lit, dim=0)
            std = torch.std(num_lit, dim=0)
            num_lit = (num_lit - mean) / std
        else:
            raise ValueError("Unkown normalization option")

        # includes all numeric literals for all entities, with the entities
        # being ordered by their index
        self.num_lit = num_lit.to(self.config.get("job.device"))

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
        return self.num_lit[indexes,:]
    
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
        # there are not parameters to penalize
        return result