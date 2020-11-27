import numpy as np
import pandas as pd
import spacy
import unicodedata

with open("../../data/fb15k-237/entity_ids.del", "r") as f:
    ent_to_idx = {ent: int(idx) for idx, ent in map(lambda s: s.strip().split("\t"), f.readlines())}

# Load raw literals
df = pd.read_csv("text_literals.txt", header=None, sep="\t")

# Load preprocessor
nlp = spacy.load('en_core_web_md')

txt_lit = np.zeros([len(ent_to_idx), 300], dtype=np.float32)
cnt = 0

for ent, txt in zip(df[0].values, df[2].values):
    key = unicodedata.normalize('NFC', ent.lower())
    idx = ent_to_idx.get(key)

    if idx is not None:
        txt_lit[idx, :] = nlp(txt).vector
    else:
        cnt += 1

print(f'Ignoring {cnt} texts.')
print('Saving text features of size {}'.format(txt_lit.shape))

np.save("text_literals.npy", txt_lit)