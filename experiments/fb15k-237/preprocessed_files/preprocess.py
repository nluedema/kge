import pandas as pd

entities_path = "/work-ceph/nluedema/kge/data/fb15k-237/entity_ids.del"
with open(entities_path, "r") as f:
    entities = list(
        map(lambda s: s.strip().split("\t")[1], f.readlines())
    )

data_path = "/work-ceph/nluedema/kge/experiments/fb15k-237/"

text_path = f"{data_path}original_files/DKRL/FB15k_mid2description.txt"
with open(text_path, "r") as f:
    text_data = list(
        map(lambda s: s.strip().split("\t"), f.readlines())
    )

# Replace "\n" with " " and remove \"
# Remove @en and surrounding quotes
# nltk.word_tokenizer transforms "..." into ``...''
# Remove duplicates
# Remove entities that are not in fb15k-237
ent_set = set()
text_path_preprocessed = f"{data_path}preprocessed_files/text_data.txt"
with open(text_path_preprocessed, "w") as f:
    for t in text_data:
        ent = t[0]
        text = t[1]
        text = text.replace("\\n", " ").replace('\\"', "")

        if text[-3:] != "@en":
            raise ValueError("Something went wrong")
        if text[0] != '"' and text[-4] != '"':
            raise ValueError("Something went wrong")
        if ent not in ent_set:
            ent_set.add(ent)
        else:
            continue
        if ent not in entities:
            continue
        f.write(f"{ent}\thasDescription\t{text[1:-4]}\n")

numeric_path = f"{data_path}original_files/KBLRN/FB15K_NumericalTriples.txt"
with open(numeric_path, "r") as f:
    numeric_data = list(
        map(lambda s: s.strip().split("\t"), f.readlines())
    )

# Remove entities that are not in fb15k-237
numeric_path_preprocessed = f"{data_path}preprocessed_files/numeric_data.txt"
with open(numeric_path_preprocessed, "w") as f:
    for t in numeric_data:
        ent = t[0]
        rel = t[1]
        val = t[2]

        if ent not in entities:
            continue
        f.write(f"{ent}\t{rel}\t{val}\n")

#numeric_path = f"{data_path}original_files/LiteralE/numerical_literals.txt"
#numeric_data = pd.read_csv(numeric_path,header=None, names=["ent","rel","val"] ,sep="\t")
#
## Remove duplicates
#numeric_data = numeric_data.drop_duplicates(
#    subset=["ent","rel"],keep="last"
#)
#
## Remove entities that are not in fb15k-237
#numeric_path_preprocessed = f"{data_path}preprocessed_files/numeric_data.txt"
#with open(numeric_path_preprocessed, "w") as f:
#    for ent, rel, val in numeric_data.values:
#        if ent not in entities:
#            continue
#        f.write(f"{ent}\t{rel}\t{val}\n")
#