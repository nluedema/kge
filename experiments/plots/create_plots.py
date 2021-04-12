import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def get_modalities(x):
    modalities = []
    if "text" in x:
        modalities.append("text")
    if "numeric" in x:
        modalities.append("num")
    return "-".join(modalities)

# load data
fb = pd.read_csv(
    "/work-ceph/nluedema/kge/local/experiments/fb15k-237/fb15k-237_all_trials.csv"
)
yago = pd.read_csv(
    "/work-ceph/nluedema/kge/local/experiments/yago3-10/yago3-10_all_trials.csv"
)
all_data = pd.concat([fb,yago])

# make sure no best trials are included
all_data = all_data[~all_data["folder"].str.contains("-best")]

# add scorer column
all_data['scorer'] = all_data['model']

# change model column
all_data['model'] = all_data['folder'].str.extract(
    r'(complex|dkrl|literale|mkbe)'
)

# change dataset column (for mkbe)
all_data['dataset'] = all_data['folder'].str.extract(
    r'(fb15k-237|yago3-10)'
)

# add modalities column
all_data['modalities'] = all_data['folder'].apply(get_modalities)

# set datasets 
datasets = ["fb15k-237", "yago3-10"]
dataset_labels = ["FB15K-237", "YAGO3-10"]

# set metric
metric = "fwt_mrr"
metric_label = "Validation MRR"

# set dataset yaxis range
dataset_yaxis_range = [(-1.24,36.22), (-2.40,51.03)]

# define plots
plot_dicts = [
    {
        "filename":"struct_params.pdf",
        "cols":["emb_e_dim","train_batch_size","train_optimizer"],
        "cols_labels":["Embedding size","Batch size", "Optimizer"],
        "cols_categories":[
            [128,256],
            [2048,4096],
            ["Adam","Adagrad"],
        ],
        "cols_categories_labels":[
            ["128","256"],
            ["2048","4096"],
            ["Adam","Adagrad"],
        ]
    },
    {
        "filename":"model_modalities.pdf",
        "cols":["model","modalities"],
        "cols_labels":["Model","Modality"],
        "cols_categories":[
            ["complex","dkrl","literale","mkbe"],
            ["text","num","text-num"]
        ],
        "cols_categories_labels":[
            ["ComplEx","DKRL","LiteralE","MKBE"],
            ["text","num","text-num"]
        ]
    },
    {
        "filename":"text_cnn.pdf",
        "cols":[
            "tcnn_dim_feature_map","tcnn_kernel_size_max_pool",
            "tcnn_kernel_size_conv","tcnn_activation"
        ],
        "cols_labels":[
            "Feature map dimension","Kernel size max pool",
            "Kernel size convolution","Activation"
        ],
        "cols_categories":[
            [40,60,80],
            [4,6,8],
            [2,3],
            ["tanh","relu"]
        ],
        "cols_categories_labels":[
            ["40","60","80"],
            ["4","6","8"],
            ["2","3"],
            ["TanH","ReLU"]
        ]
    },
    {
        "filename":"numeric_embedder.pdf",
        "cols":[
            "numeric_feature_num_layers","numeric_feature_activation",
            "numeric_mlp_num_layers","numeric_mlp_activation"
        ],
        "cols_labels":[
            "Number of layers",
            "Activation",
            "Number of layers",
            "Activation"
        ],
        "cols_categories":[
            [1,2],
            ["tanh","relu"],
            [1,2],
            ["tanh","relu"]
        ],
        "cols_categories_labels":[
            ["1","2"],
            ["TanH","ReLU"],
            ["1","2"],
            ["TanH","ReLU"]
        ]
    }
]

for plot_dict in plot_dicts:
    filename = plot_dict["filename"]
    cols = plot_dict["cols"]
    cols_labels = plot_dict["cols_labels"]
    assert len(cols) == len(cols_labels)
    cols_categories = plot_dict["cols_categories"]
    cols_categories_labels = plot_dict["cols_categories_labels"]
    assert len(cols) == len(cols_categories)
    assert len(cols) == len(cols_categories_labels)

    # create plot
    label_rotation=0
    font_size = 7.5
    num_rows = len(datasets)
    num_cols = len(cols)

    f, axes = plt.subplots(num_rows, num_cols, sharex="col", sharey="row")
    plt.subplots_adjust(hspace=0.1, wspace=0.2)
    plt.xticks(rotation=label_rotation)

    # add dataset names to subplot rows
    pad = 5
    for ax, row, axis_range in zip(
        axes[:,0], dataset_labels, dataset_yaxis_range
    ):
        ax.annotate(row, xy=(0.0,0.5), rotation=90,
                    xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size=font_size, ha='right', va='center')
        ax.set_ylim(axis_range)

    # create each column 
    for col in range(num_cols):
        current_col = cols[col]
        title = cols_labels[col]
        current_col_categories = cols_categories[col]
        current_col_categories_labels = cols_categories_labels[col]
        assert len(current_col_categories) == len(current_col_categories_labels)

        # each dataset is a row in the main plot
        for row in range(num_rows):
            current_dataset = datasets[row]

            dataset_data = all_data.loc[all_data['dataset'] == current_dataset]
            dataset_data = dataset_data.loc[
                dataset_data[current_col].isin(current_col_categories)
            ]

            # Remove the LiteralE runs that have num_layers = 0
            if current_col == "numeric_feature_activation":
                dataset_data = dataset_data.loc[
                    dataset_data["numeric_feature_num_layers"] > 0
                ]

            box = sns.boxplot(x=dataset_data[current_col],
                    y=dataset_data[metric]*100,
                    order=current_col_categories,
                    linewidth=0.5,
                    fliersize=1,
                    ax=axes[row][col]
                    )

            if row != len(datasets) - 1:
                if  "numeric_feature_" in current_col:
                    box.set_title(title, size=font_size, y=1.025)
                    box.text(
                        0.5, 1.02, "(NumericDKRLEmbedder)", transform=box.transAxes,
                        fontsize=font_size-1.25, ha='center'
                    )
                elif "numeric_mlp_" in current_col:
                    box.set_title(title, size=font_size, y=1.025)
                    box.text(
                        0.5, 1.02, "(NumericMKBEEmbedder)", transform=box.transAxes,
                        fontsize=font_size-1.25, ha='center'
                    )
                else:
                    box.set_title(title, size=font_size)

            box.tick_params(labelsize=font_size)
            if col == 0:
                box.set_ylabel(metric_label, fontsize=font_size)
            else:
                box.set_ylabel('')

            # add xticks labels when in last row
            if row == (len(datasets) - 1):
                axes[row][col].set_xticklabels(
                    current_col_categories_labels, ha='center'
                )

    # add labels to box plots
    for ax in axes.flat:
        plt.sca(ax)
        plt.xticks(rotation=label_rotation)
        ax.set(xlabel='')

    # save figure for current attribute            
    box.get_figure().savefig(
            filename, 
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.04)