with open("../../data/yago3-10/entity_ids.del","r") as f:
    ent_to_idx = {ent: int(idx) for idx, ent in map(lambda s: s.strip().split("\t"), f.readlines())}

with open("numerical_data_preprocessed","r") as f:
    numerical_triples_preprocessed = list(map(lambda s: s.strip().split("\t"), f.readlines()))

# check if the prefix "numerical_entity" can be used as numerical entity name
# without clashes with existing entities
for ent in ent_to_idx.keys():
    if "numerical_entity" in ent:
        print(ent)

# check if numerical_triples_preprocessed is always length 3
for t in numerical_triples_preprocessed:
    if len(t) != 3:
        print(t)

# count number of numerical relations
numerical_relations = set()
for t in numerical_triples_preprocessed:
    numerical_relations.add(t[1])
print(len(numerical_relations))

# count number of numerical entities, which equals number of numerical triples
print(len(numerical_triples_preprocessed))
