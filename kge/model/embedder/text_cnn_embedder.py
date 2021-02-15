from torch import Tensor
import torch

from kge import Config, Dataset
from kge.job import Job
from kge.model import KgeEmbedder

from typing import List

import gensim.downloader as api
import nltk
import string

class TextCNN(torch.nn.Module):
    def __init__(
        self,
        dim_embedding,
        dim_word,
        dim_feature_map,
        kernel_size_conv,
        kernel_size_max_pool,
        activation,
    ):
        super().__init__()
        if activation == "tanh":
            self.activation = torch.nn.Tanh()
        elif activation == "relu":
            self.activation = torch.nn.ReLU()
        else:
            raise ValueError("unkown activation function specified")

        self.conv_1 = torch.nn.Conv1d(dim_word, dim_feature_map, kernel_size_conv)
        self.conv_2 = torch.nn.Conv1d(dim_feature_map, dim_embedding, kernel_size_conv)
        self.max_pool = torch.nn.MaxPool1d(
            kernel_size=kernel_size_max_pool,
            stride=kernel_size_max_pool, # makes the pooling windows non-overlapping
            # if the last window does not fit kernel size, ceil_mode=True, does not
            # drop the last window, but instead performs max pool over the remaining
            # part of the window
            ceil_mode=True,
            )

    def forward(self, text_seq):
        output = self.conv_1(text_seq)
        output = self.max_pool(output)
        output = self.activation(output)
        output = self.conv_2(output)
        output = output.mean(dim=2) # take mean over the sequence dimension
        output = self.activation(output)
        return output

class TextCNNEmbedder(KgeEmbedder):
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

        # process text data
        # creates self.text_map and self.word_embeddings
        # text_map, maps each text sequence to a corresponding sequence of word id's
        # word_embeddings, maps each word id to a corresponding word embedding
        # also set self.dim_word
        self.process_text()

        # initialize text cnn
        self.text_cnn = TextCNN(
            dim_embedding=self.dim,
            dim_word=self.dim_word,
            dim_feature_map=self.get_option("dim_feature_map"),
            kernel_size_conv=self.get_option("kernel_size_conv"),
            kernel_size_max_pool=self.get_option("kernel_size_max_pool"),
            activation=self.get_option("activation"),
        )

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

    def process_text(self):
        # load text triples
        with open(self.filename,"r") as f:
            data = list(
                map(lambda s: s.strip().split("\t"), f.readlines())
            )

        # load word embeddings
        gensim_model_name = self.get_option("gensim_model_name")
        if "glove-wiki-gigaword" in gensim_model_name:
            self.punctuation_flag=True
            self.stopwords_flag=True
            self.lower_flag=True
        elif "google-news" in gensim_model_name:
            self.punctuation_flag=True
            self.stopwords_flag=True
            self.lower_flag=False
        else:
            raise ValueError(
                f"No preprocessing for {gensim_model_name} available"
            )
        self.model = api.load(gensim_model_name)

        # CAREFUL
        # returns a list of word lists
        # each word list corresponds to a text description
        # the text descriptions are NOT ordered as for dkrl or literale
        # instead they are ordered based on their multimodal entity id
        text_data = self.preprocess_text(data)

        # build word map
        word_map = {}
        word_map["<padding>"] = 0

        max_len = 0

        for text in text_data:
            if len(text) > max_len:
                max_len = len(text)
            for w in text:
                if w not in word_map:
                    word_map[w] = len(word_map)
        
        # restrict sequence length, if max_sequence length is set
        max_sequence_length = self.get_option("max_sequence_length")
        if (max_sequence_length > -1) and (max_sequence_length < max_len):
            max_len = max_sequence_length

        # build text map 
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

        self.text_map = self.text_map.to(self.config.get("job.device"))

        # load word embeddings into tensor
        self.dim_word = self.model.vector_size
        word_embeddings = torch.zeros(
            (len(word_map),self.dim_word),dtype=torch.float32
        )
        for w,w_id in word_map.items():
            if w == "<padding>":
                w_embedding = torch.zeros(self.dim_word,dtype=torch.float32)
            elif w in self.model:
                w_embedding = torch.from_numpy(self.model[w].copy())
            else:
                # TODO: deal with out of vocablary words
                # currently there are no OOV words, because _preprocess_text
                # filters out all words that are not in model
                pass

            word_embeddings[w_id,:] = w_embedding

        self.word_embeddings = torch.nn.Embedding.from_pretrained(
            word_embeddings,freeze=self.get_option("freeze_word_embeddings"),
            padding_idx=0
        )
    
    def preprocess_text(self,data):
        # CAREFULL, DOUBLE CHECK IF t[1] OR t[2]
        # SHOULD BE THE SAME FOR ALL DATASETS
        text_data = []
        for t in data:
            text = t[2]
            text_data.append(
                self._preprocess_text(text=text)
            )
        return text_data

    def _preprocess_text(
        self,text
    ):
        punctuation = list(string.punctuation)
        stopwords = set(nltk.corpus.stopwords.words('english'))

        sents = nltk.tokenize.sent_tokenize(text)
        final = []
        for sent in sents:
            if self.lower_flag:
                sent = sent.lower()

            sent = nltk.tokenize.word_tokenize(sent)
            for w in sent:
                punctuation_bool = True
                stopwords_bool = True

                if self.punctuation_flag:
                    punctuation_bool = w not in punctuation

                if self.stopwords_flag:
                    stopwords_bool = w not in stopwords

                if punctuation_bool and stopwords_bool and w in self.model:
                    final.append(w)

        return final
    
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