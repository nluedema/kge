import numpy as np
import pandas as pd

with open("../../data/fb15k-237/entity_ids.del", "r") as f:
    ent_to_idx = {ent: int(idx) for idx, ent in map(lambda s: s.strip().split("\t"), f.readlines())}

# Load raw literals
df = pd.read_csv("numerical_literals.txt", header=None, sep="\t")

rel_to_idx = {v: k for k, v in enumerate(df[1].unique())}

# Resulting file
num_lit = np.zeros([len(ent_to_idx), len(rel_to_idx)], dtype=np.float32)

# Create literal wrt vocab
for s, p, lit in df.values:
    try:
        num_lit[ent_to_idx[s.lower()], rel_to_idx[p]] = lit
    except KeyError:
        print(f"Problem for:{s}\t{p}\t{lit}")
        continue

np.save("numerical_literals.npy", num_lit)
