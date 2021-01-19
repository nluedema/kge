import gensim
import gensim.downloader as api
import nltk
import string
import pandas as pd

glove = api.load("glove-wiki-gigaword-300")

with open(
    "/work-ceph/nluedema/kge/data/yago3-10-mkbe/text_description.txt"
) as f:
    data = list(
        map(lambda s: s.strip().split("\t"), f.readlines())
    )

punctuation = list(string.punctuation)
stopwords = set(nltk.corpus.stopwords.words('english'))

text_data = []
for t in data:
    sents = nltk.tokenize.sent_tokenize(t[1])
    final = []
    for sent in sents:
        sent = sent.lower()
        sent = nltk.tokenize.word_tokenize(sent)
        for w in sent:
            if w not in stopwords and w not in punctuation and w in glove:
                final.append(w)
    text_data.append(final)

desc_len = []
for desc in text_data:
    desc_len.append(len(desc))
desc_len = pd.Series(desc_len)
sum(desc_len < 3)
    
vocab = {}
for sent in text_data:
    for word in sent:
        try:
            vocab[word] += 1
        except KeyError:
            vocab[word] = 1

vocab_sort = {
    k: v for k, v in sorted(vocab.items(), key=lambda item: item[1])
}
print({k: vocab_sort[k] for k in list(vocab_sort)[-50:]})


