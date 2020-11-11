import pandas as pd
experiments_dir = "/home/nluedema/kge/local/experiments/"
five_times_dir = experiments_dir + "20201031-013659-literale-train-5-times/trace_dump.csv"
search_ns_kl_dir = experiments_dir + "20201030-162925-search-distmult-literale-negative_sampling-kl/trace_dump.csv"
search_1vsAll_kl_dir = experiments_dir + "20201031-153811-search-distmult-literale-1vsAll-kl/trace_dump.csv"
search_KvsAll_kl_dir = experiments_dir + "20201103-181723-search-distmult-literale-KvsAll-kl/trace_dump.csv"

iclr2020_fb15k_237_all_trials_dir = "/home/nluedema/kge-iclr20/data_dumps/iclr2020-fb15k-237-all-trials.csv"

five_times = pd.read_csv(five_times_dir)
search_ns_kl = pd.read_csv(search_ns_kl_dir)
search_1vsAll_kl = pd.read_csv(search_1vsAll_kl_dir)
search_KvsAll_kl = pd.read_csv(search_KvsAll_kl_dir)

iclr2020_fb15k_237_all_trials = pd.read_csv(iclr2020_fb15k_237_all_trials_dir)

iclr2020_fb15k_237_all_trials_ns_kl_top_5 = iclr2020_fb15k_237_all_trials.query(
    """folder == '/work/pi1/iclr2020/fb15k-237/distmult-negative_sampling-kl'"""
    )["fwt_mrr"].sort_values(ascending=False)[0:5].index
search_ns_kl_top_5 = search_ns_kl["fwt_mrr"].sort_values(ascending=False)[0:5].index

iclr2020_fb15k_237_all_trials.loc[
    iclr2020_fb15k_237_all_trials_ns_kl_top_5,
    ["fwt_mrr", "fwt_hits@1", "fwt_hits@3", "fwt_hits@10"]]
search_ns_kl.loc[
    search_ns_kl_top_5,
    ["fwt_mrr", "fwt_hits@1", "fwt_hits@3", "fwt_hits@10"]]

iclr2020_fb15k_237_all_trials_1vsAll_kl_top_5 = iclr2020_fb15k_237_all_trials.query(
    """folder == '/work/pi1/iclr2020/fb15k-237/distmult-1vsAll-kl'"""
    )["fwt_mrr"].sort_values(ascending=False)[0:5].index
search_1vsAll_kl_top_5 = search_1vsAll_kl["fwt_mrr"].sort_values(ascending=False)[0:5].index

iclr2020_fb15k_237_all_trials.loc[
    iclr2020_fb15k_237_all_trials_1vsAll_kl_top_5,
    ["fwt_mrr", "fwt_hits@1", "fwt_hits@3", "fwt_hits@10"]]
search_1vsAll_kl.loc[
    search_1vsAll_kl_top_5,
    ["fwt_mrr", "fwt_hits@1", "fwt_hits@3", "fwt_hits@10"]]

iclr2020_fb15k_237_all_trials_KvsAll_kl_top_5 = iclr2020_fb15k_237_all_trials.query(
    """folder == '/work/pi1/iclr2020/fb15k-237/distmult-KvsAll-kl'"""
    )["fwt_mrr"].sort_values(ascending=False)[0:5].index
search_KvsAll_kl_top_5 = search_KvsAll_kl["fwt_mrr"].sort_values(ascending=False)[0:5].index

iclr2020_fb15k_237_all_trials.loc[
    iclr2020_fb15k_237_all_trials_KvsAll_kl_top_5,
    ["fwt_mrr", "fwt_hits@1", "fwt_hits@3", "fwt_hits@10"]]
search_KvsAll_kl.loc[
    search_KvsAll_kl_top_5,
    ["fwt_mrr", "fwt_hits@1", "fwt_hits@3", "fwt_hits@10"]]



search_ns_kl.loc[
    search_ns_kl_top_5,
    ["fwt_mrr", "emb_e_dropout", "emb_r_dropout", "gate_dropout"]
]
search_ns_kl.loc[
    search_ns_kl_top_5,
    ["fwt_mrr", "emb_regularize", "emb_regularize_p", "emb_regularize_weighted",
    "emb_e_regularize_weight", "emb_r_regularize_weight",
    "gate_regularize", "gate_regularize_p", "gate_regularize_weight"]
]
search_ns_kl.loc[
    search_ns_kl_top_5,
    ["fwt_mrr", "train_optimizer", "train_lr", "train_lr_scheduler_patience"]
]

search_1vsAll_kl.loc[
    search_1vsAll_kl_top_5,
    ["fwt_mrr", "emb_e_dropout", "emb_r_dropout", "gate_dropout"]
]
search_1vsAll_kl.loc[
    search_1vsAll_kl_top_5,
    ["fwt_mrr", "emb_regularize", "emb_regularize_p", "emb_regularize_weighted",
    "emb_e_regularize_weight", "emb_r_regularize_weight",
    "gate_regularize", "gate_regularize_p", "gate_regularize_weight"]
]
search_1vsAll_kl.loc[
    search_1vsAll_kl_top_5,
    ["fwt_mrr", "train_optimizer", "train_lr", "train_lr_scheduler_patience"]
]