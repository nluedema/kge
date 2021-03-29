import pandas as pd
import numpy as np
entities_path = "/work-ceph/nluedema/kge/data/fb15k-237/entity_ids.del"
kblrn_path = "/work-ceph/nluedema/kge/experiments/fb15k-237/original_files/KBLRN/FB15K_NumericalTriples.txt"
literale_path = "/work-ceph/nluedema/kge/experiments/fb15k-237/original_files/LiteralE/numerical_literals.txt"

entities = pd.read_csv(entities_path, header=None, names=["id","ent"], sep="\t")

kblrn = pd.read_csv(kblrn_path,header=None, names=["ent","rel","value"] ,sep="\t")
# remove surrouding <> to equalize kblrn and literale rel naming
kblrn["rel"] = kblrn["rel"].apply(lambda x: x[1:-1])

literale = pd.read_csv(literale_path,header=None, names=["ent","rel","value"] ,sep="\t")

len(kblrn["ent"].unique())
# 12493
len(literale["ent"].unique())
# 9941

len(kblrn["rel"].unique())
# 116
len(literale["rel"].unique())
# 121

len(kblrn)
# 29395
sum(~kblrn[["ent","rel"]].duplicated())
# 29395
len(literale)
# 70257
sum(~literale[["ent","rel"]].duplicated())
# 18786
literale_deduplicated = literale[~literale[["ent","rel"]].duplicated()]

len(literale_deduplicated[np.in1d(literale_deduplicated["ent"], entities["ent"])])
# 18671
len(kblrn[np.in1d(kblrn["ent"], entities["ent"])])
# 29247

kblrn

kblrn["rel"].value_counts()[0:20]