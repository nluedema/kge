import torch
from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel

import importlib

class DKRLScorer(RelationalScorer):

    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key=None
    ):
        super().__init__(config, dataset, configuration_key)
        try:
            base_scorer_class_name = self.get_option("base_scorer.class_name")
            base_scorer_type = self.get_option("base_scorer.type")
            module = importlib.import_module(f"kge.model.{base_scorer_type}")
        except:
            raise Exception(f"Can't find {configuration_key}.base_scorer in config")

        try:
            self.base_scorer = getattr(module, base_scorer_class_name)(
                config=config,
                dataset=dataset,
                configuration_key=configuration_key + ".base_scorer"
            )
        except:
            raise Exception(f"Can't find {base_scorer_class_name}")

    def score_emb(self, s_emb, p_emb, o_emb, combine: str):
        s_emb_struct = s_emb[:,:,0]
        o_emb_struct = o_emb[:,:,0]

        # calculate structural score
        score = self.base_scorer.score_emb(
            s_emb_struct, p_emb, o_emb_struct, combine
        )

        # add multimodal scores
        for i,modality in enumerate(self.get_option("entity_embedder.modalities")):
            if modality != "struct": 
                s_emb_multimodal = s_emb[:,:,i]
                o_emb_multimodal = o_emb[:,:,i]

                # score mm
                score_multimodal = self.base_scorer.score_emb(
                    s_emb_multimodal, p_emb, o_emb_multimodal, combine
                )
                #score ms
                score_multimodal += self.base_scorer.score_emb(
                    s_emb_multimodal, p_emb, o_emb_struct, combine
                )
                #score sm
                score_multimodal += self.base_scorer.score_emb(
                    s_emb_struct, p_emb, o_emb_multimodal, combine
                )
                modality_weight = self.config.get(
                    f"train.multimodal_args.{modality}.weight"
                )
                score += (modality_weight * score_multimodal)

        return score

class DKRLModel(KgeModel):

    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key=None,
        init_for_load_only=False,
    ):
        super().__init__(
            config=config,
            dataset=dataset,
            scorer=DKRLScorer,
            configuration_key=configuration_key,
            init_for_load_only=init_for_load_only,
        )

        if self.get_option("entity_embedder.type") != "dkrl_embedder":
            raise ValueError("dkrl_model only works with dkrl_embedder")