import pandas as pd
import numpy as np
literale_path = "/work-ceph/nluedema/kge/experiments/fb15k-237/original_files/LiteralE/text_literals.txt"
dkrl_path = "/work-ceph/nluedema/kge/experiments/fb15k-237/original_files/DKRL/FB15k_mid2description.txt"

dkrl = pd.read_csv(dkrl_path,header=None, names=["ent","text"] ,sep="\t")
literale = pd.read_csv(literale_path,header=None, names=["ent","rel","text"] ,sep="\t")

# load entities
entities_path = "/work-ceph/nluedema/kge/data/fb15k-237/entity_ids.del"
entities = pd.read_csv(entities_path,header=None,names=["id","ent"],sep="\t")

len(entities)
# 14507
sum(np.in1d(entities["ent"],literale["ent"]))
# 14515
sum(np.in1d(entities["ent"],dkrl["ent"]))