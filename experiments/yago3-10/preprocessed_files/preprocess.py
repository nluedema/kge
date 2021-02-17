import pandas as pd

entities_path = "/work-ceph/nluedema/kge/data/yago3-10/entity_ids.del"
with open(entities_path, "r") as f:
    entities = list(
        map(lambda s: s.strip().split("\t")[1], f.readlines())
    )

data_path = "/work-ceph/nluedema/kge/experiments/yago3-10/"

text_path = f"{data_path}original_files/Textual data.txt"
with open(text_path, "r") as f:
    text_data = list(
        map(lambda s: s.strip().split("\t"), f.readlines())
    )

# Remove duplicates
# Remove entities that are not in yago3-10
ent_set = set()
text_path_preprocessed = f"{data_path}preprocessed_files/text_data.txt"
with open(text_path_preprocessed, "w") as f:
    for t in text_data:
        ent = t[0]
        text = t[1]
        # deal with unicode in text description
        text = text.encode("utf-8").decode("unicode-escape")
        # remove newlines that were introduced by decoding
        text = text.replace("\n", " ")

        if ent not in ent_set:
            ent_set.add(ent)
        else:
            continue

        if ent not in entities:
            continue

        f.write(f"{ent}\thasDescription\t{text}\n")

numeric_path = f"{data_path}original_files/numerical_data_merged.txt"
with open(numeric_path, "r") as f:
    numeric_data = list(
        map(lambda s: s.strip().split("\t"), f.readlines())
    )

# Remove entities that are not in yago3-10
# Count the occurences of each relation
numeric_data_removed = []
rel_counts = {}
for t in numeric_data:
    ent = t[0]
    rel = t[1]

    if ent not in entities:
        continue
    numeric_data_removed.append(t)

    if rel not in rel_counts:
        rel_counts[rel] = 1
    else:
        rel_counts[rel] += 1

# Remove triples with relations that have less than 5 occurences
numeric_path_preprocessed = f"{data_path}preprocessed_files/numeric_data.txt"
with open(numeric_path_preprocessed, "w") as f:
    for t in numeric_data_removed:
        ent = t[0]
        rel = t[1]
        val = t[2]

        if rel_counts[rel] < 5:
            continue
        f.write(f"{ent}\t{rel}\t{val}\n")
