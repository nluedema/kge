import pandas as pd
import numpy as np
data_path = "/work-ceph/nluedema/kge/experiments/yago3-10/"
numeric = pd.read_csv(
    f"{data_path}original_files/numerical_data_merged.txt",
    names=["ent","rel","val"], header=None,sep="\t"
)

numeric[["ent","rel"]].duplicated().sum()
# => no duplicates

numeric["val"].apply(lambda x: len(str(x))).value_counts()

text = pd.read_csv(
    f"{data_path}original_files/Textual data.txt",
    names=["ent","desc"], header=None,sep="\t"
)
text.duplicated().sum()
# => 7 duplicates

entities_path = "/work-ceph/nluedema/kge/data/yago3-10/entity_ids.del"
entities = pd.read_csv(entities_path, header=None, names=["id","ent"], sep="\t")
text[np.in1d(text["ent"], entities["ent"])].duplicated().sum()
len(text)
len(text[np.in1d(text["ent"], entities["ent"])])

