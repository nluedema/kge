from torch import Tensor
import torch

from kge import Config, Dataset
from kge.job import Job
from kge.model import KgeEmbedder
from kge.misc import round_to_points

from typing import List

import gensim.downloader as api

class TextCNN(torch.nn.Module):
    def __init__(
        self,
        dim_embedding,
        dim_word,
        kernel_size
    ):
        super().__init__()

        self.conv_1 = torch.nn.Conv1d(dim_word,dim_embedding,kernel_size)
        self.conv_2 = torch.nn.Conv1d(dim_embedding,dim_embedding,kernel_size)
        self.relu = torch.nn.ReLU()

    def forward(self, text_seq):
        conv_1 = self.relu(self.conv_1(text_seq))
        conv_2 = self.relu(self.conv_2(conv_1))

        # take mean over the sequence dimension
        return conv_2.mean(dim=2)

class TextEmbedder(KgeEmbedder):
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

        # load text triples
        with open(self.filename,"r") as f:
            data = list(
                map(lambda s: s.strip().split("\t"), f.readlines())
            )
        
        # build word map
        word_map = {}
        word_map["<padding>"] = 0

        max_len = 0

        text_data = []
        for t in data:
            text = t[2].split(" ")
            text_data.append(text)
            if len(text) > max_len:
                max_len = len(text)
            for w in text:
                if w not in word_map:
                    word_map[w] = len(word_map)
        
        # build text to id
        self.text_map = torch.zeros(
            (len(text_data),max_len),dtype=torch.long,requires_grad=False
        )
        for i,text in enumerate(text_data):
            word_to_id = []
            for j in range(0,max_len):
                if j < len(text):
                    word_to_id.append(word_map[text[j]])
                else:
                    word_to_id.append(word_map["<padding>"])
            
            word_to_id = torch.tensor(word_to_id,dtype=torch.long)
            self.text_map[i,:] = word_to_id
        
        self.text_map = self.text_map.to(config.get("job.device"))
        

        # load word embeddings 
        glove = api.load("glove-wiki-gigaword-300")
        word_embeddings = torch.empty(
            (len(word_map),300),dtype=torch.float32
        )
        for w,w_id in word_map.items():
            if w == "<padding>":
                w_embedding = torch.zeros(300,dtype=torch.float32)
            elif w in glove:
                w_embedding = torch.from_numpy(glove[w].copy())
            else:
                # check if word is in glove
                # if not initialize word embedding as config describes
                # if config says only known words produce error
                if self.get_option("missing_words") != "random":
                    raise ValueError(f"no word embedding found for {2}")

            word_embeddings[w_id,:] = w_embedding
        
        self.word_embeddings = torch.nn.Embedding.from_pretrained(
            word_embeddings,freeze=False,padding_idx=0
        )
        
        #self.word_embeddings.to(config.get("job.device"))

        #test = [27,33333,107261]
        #text_map[test,:].shape
        #word_embeddings(text_map[test,:]).shape
        #word_embeddings(text_map[test,:]).permute(0,2,1).shape

        #word_embeddings(text_map[test,:])[1,1,:] == word_embeddings(text_map[test,:]).permute(0,2,1)[1,:,1]

        # initialize text cnn
        self.text_cnn = TextCNN(self.dim, 300, kernel_size=4)

        if not init_for_load_only:
            # initialize weights
            for name, weights in self.text_cnn.named_parameters():
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
            # get sequence of word indexes for each text description
            # N*L, with batch size N and sequence length L
            word_indexes = self.text_map[indexes,:]

            # get sequence of word embeddigs for each text description
            # N*L*C, with number of channels C
            text_seq = self.word_embeddings(word_indexes)
            # do permute to get from N*L*C to N*C*L, which is expected by conv1d
            text_seq = text_seq.permute(0,2,1)

            return self.text_cnn(text_seq)
    
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
            for name, parameter in self.text_cnn.named_parameters():
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