import gensim.downloader as api
import nltk
import string
import numpy as np

def _preprocess_text(
    text, model, lower_flag, punctuation_flag, stopwords_flag,
    with_oov
):
    punctuation = list(string.punctuation)
    stopwords = set(nltk.corpus.stopwords.words('english'))
    
    sents = nltk.tokenize.sent_tokenize(text)
    final = []
    for sent in sents:
        if lower_flag:
            sent = sent.lower()

        sent = nltk.tokenize.word_tokenize(sent)
        for w in sent:
            punctuation_bool = True
            stopwords_bool = True

            if punctuation_flag:
                punctuation_bool = w not in punctuation

            if stopwords_flag:
                stopwords_bool = w not in stopwords

            if with_oov:
                if punctuation_bool and stopwords_bool:
                    final.append(w)
            else:
                if punctuation_bool and stopwords_bool and w in model:
                    final.append(w)

    return final

def get_words_lengths(text_data):
    lengths = []
    words = set()
    for text in text_data:
        lengths.append(len(text))
        for w in text:
            words.add(w)
    return words,lengths

entities_path = "/work-ceph/nluedema/kge/data/fb15k-237/entity_ids.del"
with open(entities_path, "r") as f:
    entities = list(
        map(lambda s: s.strip().split("\t")[1], f.readlines())
    )

text_path ="/work-ceph/nluedema/kge/experiments/fb15k-237/preprocessed_files/text_data.txt"
with open(text_path,"r") as f:
    data = list(
        map(lambda s: s.strip().split("\t"), f.readlines())
    )

gensim_model_name = "glove-wiki-gigaword-100"
model = api.load(gensim_model_name)

data_index = {t[0]: t[2] for t in data}

text_data = []
for entity in entities:
    if entity in data_index:
        text = _preprocess_text(
            text=data_index[entity], model=model, punctuation_flag=True,
            stopwords_flag=True, lower_flag=True, with_oov=False
        )
    else:
        text = []
    text_data.append(text)

text_data_with_oov = []
for entity in entities:
    if entity in data_index:
        text = _preprocess_text(
            text=data_index[entity], model=model, punctuation_flag=True,
            stopwords_flag=True, lower_flag=True, with_oov=True
        )
    else:
        text = []
    text_data_with_oov.append(text)

words,lengths = get_words_lengths(text_data)
words_with_oov,lengths_with_oov = get_words_lengths(text_data_with_oov)

len(words)
# 62869
len(words_with_oov)
# 78013

oov_words = []
for word in words_with_oov:
    if word not in words:
        oov_words.append(word)

lengths = np.array(lengths)
lengths_with_oov = np.array(lengths_with_oov)

lengths.max()
lengths_with_oov.max()

len(lengths)
# 14541

sum(lengths > 300)
# 6
sum(lengths > 250)
# 17
sum(lengths > 200)
# 142