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
    words = {}
    for text in text_data:
        lengths.append(len(text))
        for w in text:
            if w in words:
                words[w] += 1
            else:
                words[w] = 1
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
        text_data.append(text)
    #else:
    #    text = []
    #    text_data.append(text)

text_data_with_oov = []
for entity in entities:
    if entity in data_index:
        text = _preprocess_text(
            text=data_index[entity], model=model, punctuation_flag=True,
            stopwords_flag=True, lower_flag=True, with_oov=True
        )
        text_data_with_oov.append(text)
    #else:
    #    text = []
    #    text_data_with_oov.append(text)

words,lengths = get_words_lengths(text_data)
words_with_oov,lengths_with_oov = get_words_lengths(text_data_with_oov)

len(words)
# 62869
len(words_with_oov)
# 78013
sum(lengths)
# 1235440
sum(lengths_with_oov)
# 1253459

oov_words = [w for w in words_with_oov if w not in words]
len(oov_words)
# 15144
sum([words_with_oov[w] for w in oov_words])
# 18019

# 18019/1253459 = 0.0144
# 1.44% of the preprocessed words are oov

oov_words[0:40]
oov_words[-40:]

dashes = [chr(45), chr(8208), chr(8209), chr(8210), chr(8211), chr(8212), chr(8213)]
oov_words_dashes = [w for w in oov_words if any(dash in w for dash in dashes)]
len(oov_words_dashes)
# 5255
sum([words_with_oov[w] for w in oov_words_dashes])
# 6845
oov_words_dashes[0:40]

words_dashes = ([w for w in words if any(dash in w for dash in dashes)])
len(words_dashes)
# 4175
sum([words[w] for w in words_dashes])
# 17149
words_dashes[0:40]

import re
def split_dash(w):
    return re.split(
        f"[{chr(45)}{chr(8208)}{chr(8209)}{chr(8210)}{chr(8211)}{chr(8212)}{chr(8213)}]",
        w
    )

oov_words_dashes_splits = []
for w in oov_words_dashes:
    oov_words_dashes_splits.extend(split_dash(w))
# get unique values
oov_words_dashes_splits = list(set(oov_words_dashes_splits))
len(oov_words_dashes_splits)
# 5312
oov_words_dashes_splits[0:40]

oov_words_dashes_splits_model =  [w for w in oov_words_dashes_splits if w in model]
len(oov_words_dashes_splits_model)
# 5022
oov_words_dashes_splits_model[0:40]

lengths = np.array(lengths)
lengths_with_oov = np.array(lengths_with_oov)

lengths.max()
# 403
lengths_with_oov.max()
# 417

len(lengths)
# 14515

sum(lengths > 300)
# 6
sum(lengths > 250)
# 17
sum(lengths > 200)
# 142

np.percentile(lengths, [25,50,75])
# array([ 43.,  81., 123.])
np.percentile(lengths, [10,20,30,40,50,60,70,80,90])
# array([ 23.,  36.,  50.,  64.,  81., 100., 117., 131., 150.])
np.percentile(lengths[lengths <= 250], [25,50,75])
# array([ 42.,  81., 123.])
np.percentile(lengths[lengths <= 250], [10,20,30,40,50,60,70,80,90])
# array([ 23.,  36.,  49.,  64.,  81., 100., 117., 131., 149.])

import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'axes.linewidth': 0.9})
fig = sns.displot(
    lengths, bins=10 
)
fig.set(xlabel="# of tokens", title="FB15K-237")
fig.savefig('/work-ceph/nluedema/kge/experiments/plots/fb15k_237_desc_length_dist.png')
