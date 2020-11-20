import numpy as np

# load original yago3-10 train triples and save all entities
entities_in_train = set()
with open("../../data/yago3-10/train.txt","r") as f:
   train_triples = list(map(lambda s: s.strip().split("\t"), f.readlines()))
   for t in train_triples:
        entities_in_train.add(t[0])
        entities_in_train.add(t[2])

# load numerical triples and only store those triples whose subjects are
# in entities_in_train
# mkbe provides the numerical triples split into test and train, so two files have
# to be loaded
numerical_triples_train = "numerical_data_train.txt"
numerical_triples_test = "numerical_data_test.txt"
numerical_triples_all = [numerical_triples_train, numerical_triples_test]

numerical_values = []

with open("numerical_data_preprocessed","w") as f_write:
    i = 0
    for numerical_triples in numerical_triples_all:
        with open(numerical_triples, "r") as f_read:
            for t in list(map(lambda s: s.strip().split("\t"), f_read.readlines())):
                if t[0] in entities_in_train:
                    f_write.write(
                        str(t[0])
                        + "\t"
                        + str(t[1])
                        + "\t"
                        + "numerical_entity_" + str(i)
                        + "\n"
                    )
                    i += 1
                    numerical_values.append(np.float32(t[2]))

numerical_values = np.array(numerical_values)

# normalize numerical values
numerical_values = numerical_values - np.mean(numerical_values)
numerical_values = numerical_values / np.std(numerical_values)

np.save("numerical_data_preprocessed.npy", numerical_values)




