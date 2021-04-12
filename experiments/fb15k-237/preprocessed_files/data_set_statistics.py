import pandas as pd
import csv
#FB15K-237
entities_path = "/work-ceph/nluedema/kge/data/fb15k-237/entity_ids.del"
with open(entities_path, "r") as f:
    entities = list(
        map(lambda s: s.strip().split("\t")[1], f.readlines())
    )

relations_path = "/work-ceph/nluedema/kge/data/fb15k-237/relation_ids.del"
with open(relations_path, "r") as f:
    relations = list(
        map(lambda s: s.strip().split("\t")[1], f.readlines())
    )

train_path = "/work-ceph/nluedema/kge/data/fb15k-237/train.del"
with open(train_path, "r") as f:
    train = list(
        map(lambda s: s.strip().split("\t")[1], f.readlines())
    )

valid_path = "/work-ceph/nluedema/kge/data/fb15k-237/valid.del"
with open(valid_path, "r") as f:
    valid = list(
        map(lambda s: s.strip().split("\t")[1], f.readlines())
    )

test_path = "/work-ceph/nluedema/kge/data/fb15k-237/test.del"
with open(test_path, "r") as f:
    test = list(
        map(lambda s: s.strip().split("\t")[1], f.readlines())
    )

data_path = "/work-ceph/nluedema/kge/experiments/fb15k-237/"
text_path = f"{data_path}original_files/DKRL/FB15k_mid2description.txt"
text_data = pd.read_csv(
    text_path ,sep="\t", header=None, quoting=csv.QUOTE_NONE,
    names =["ent","text"]
)

text_path_preprocessed = f"{data_path}preprocessed_files/text_data.txt"
text_data_preprocessed = pd.read_csv(
    text_path_preprocessed,sep="\t", header=None, quoting=csv.QUOTE_NONE,
    names =["ent","rel","text"]
)

numeric_path = f"{data_path}original_files/KBLRN/FB15K_NumericalTriples.txt"
numeric_data = pd.read_csv(
    numeric_path ,sep="\t", header=None, quoting=csv.QUOTE_NONE,
    names =["ent","rel","value"]
)

numeric_path_preprocessed = f"{data_path}preprocessed_files/numeric_data.txt"
numeric_data_preprocessed = pd.read_csv(
    numeric_path_preprocessed,sep="\t", header=None, quoting=csv.QUOTE_NONE,
    names =["ent","rel","value"]
)

len(entities)
len(relations)

len(text_data_preprocessed)
len(numeric_data_preprocessed)
len(numeric_data_preprocessed["rel"].unique())

len(train)
len(valid)
len(test)
len(train) + len(valid) + len(test)

len(text_data)
len(text_data_preprocessed)
len(numeric_data)
len(numeric_data_preprocessed)

len(text_data_preprocessed["rel"].unique())
len(numeric_data_preprocessed["rel"].unique())
