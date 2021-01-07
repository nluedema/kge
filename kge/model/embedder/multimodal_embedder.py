from torch import Tensor
import torch.nn
import torch.nn.functional

from kge import Config, Dataset
from kge.job import Job
from kge.model import KgeEmbedder
from kge.misc import round_to_points
from kge.misc import kge_base_dir

from typing import List


class MultimodalEmbedder(KgeEmbedder):
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

        round_embedder_dim_to = self.get_option("round_dim_to")
        if len(round_embedder_dim_to) > 0:
            self.dim = round_to_points(round_embedder_dim_to, self.dim)

        # load dataset yaml
        import yaml
        with open(f"{self.dataset.folder}/dataset.yaml", "r") as f:
            self.dataset_yaml = yaml.load(f, Loader=yaml.SafeLoader)["dataset"]

        # create embedder for each modality
        self.embedder = {}
        prefix = "files.train.modality"
        for modality in self.config.get("train.multimodal_args.modalities"):
            entity_start_idx = self.dataset_yaml[
                f"{prefix}.{modality}.entity.start_idx"
            ]
            entity_end_idx = self.dataset_yaml[
                f"{prefix}.{modality}.entity.end_idx"
            ]
            vocab_size = entity_end_idx - entity_start_idx

            if modality != "struct":
                filename = self.dataset_yaml[f"{prefix}.{modality}.filename"]
                filename = f"{self.dataset.folder}/{filename}"

                # set filename of modality embedder
                config.set(
                    f"{self.configuration_key}.{modality}.filename",
                    filename
                )

                # set start_idx of modality embedder
                config.set(
                    f"{self.configuration_key}.{modality}.start_idx",
                    entity_start_idx
                )


            # set dimension of modality embedder to dimension of parent embedder
            config.set(
                f"{self.configuration_key}.{modality}.dim",
                self.dim
            )

            embedder = KgeEmbedder.create(
                config, dataset, f"{self.configuration_key}.{modality}",
                vocab_size = vocab_size
            )

            self.embedder[modality] = embedder
        
        # HACK
        # kwargs["indexes"] is set to None, if the multimodal_embedder has
        # regularize_args.weighted set to False.
        # If the child_embedder has regularize_args.weighted set to True,
        # it tries to access kwargs["indexes"], which leads to an error

        # Set regularize_args.weighted to True
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
        from kge.job import TrainingJob

        super().prepare_job(job, **kwargs)
        for modality in self.config.get("train.multimodal_args.modalities"):
            self.embedder[modality].prepare_job(job, **kwargs)

    def embed(self, indexes: Tensor, **kwargs) -> Tensor:
        modality = kwargs["modality"]
        return self._postprocess(self.embedder[modality].embed(indexes))

    def embed_all(self, **kwargs) -> Tensor:
        raise ValueError("Should never be called, in multimodal setting")

    def _postprocess(self, embeddings: Tensor) -> Tensor:
        if self.dropout.p > 0:
            embeddings = self.dropout(embeddings)
        return embeddings
    
    def penalty(self, **kwargs) -> List[Tensor]:
        result = super().penalty(**kwargs)
        for modality in self.config.get("train.multimodal_args.modalities"):
            result += self.embedder[modality].penalty(**kwargs)
        return result