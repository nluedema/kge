import pandas as pd
data_path = "/work-ceph/nluedema/kge/experiments/fb15k-237/"

numeric = pd.read_csv(
    f"{data_path}original_files/KBLRN/FB15K_NumericalTriples.txt",
    names=["ent","rel","val"], header=None,sep="\t"
)
numeric[["ent","rel"]].duplicated().sum()
# => 0 duplicates

#numeric = pd.read_csv(
#    f"{data_path}original_files/LiteralE/numerical_literals.txt",
#    names=["ent","rel","val"], header=None,sep="\t"
#)
#numeric[["ent","rel"]].duplicated().sum()
## => 51471 duplicates 

text = pd.read_csv(
    f"{data_path}original_files/DKRL/FB15k_mid2description.txt",
    names=["ent","desc"], header=None,sep="\t"
)
text["ent"].duplicated().sum()
# => 0 duplicates
