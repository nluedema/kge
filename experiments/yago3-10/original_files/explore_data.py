import pandas as pd
data_path = "/work-ceph/nluedema/kge/experiments/yago3-10/"
numeric = pd.read_csv(
    f"{data_path}original_files/numerical_data_merged.txt",
    names=["ent","rel","val"], header=None,sep="\t"
)

numeric[["ent","rel"]].duplicated().sum()
# => no duplicates

text = pd.read_csv(
    f"{data_path}original_files/Textual data.txt",
    names=["ent","desc"], header=None,sep="\t"
)
text.duplicated().sum()
# => 7 duplicates
