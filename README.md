This repository forks [LibKGE](https://github.com/uma-pi1/kge) and implements the multimodal Knowledge Graph Embedding (mKGE) models
[DKRL](https://ojs.aaai.org/index.php/AAAI/article/view/10329),
[LiteralE](https://link.springer.com/chapter/10.1007/978-3-030-30793-6_20) and
[MKBE](https://www.aclweb.org/anthology/D18-1359/) on top of it.
DKRL and LiteralE are implemented in the master branch. MKBE is implemented in the [MKBE](https://github.com/nluedema/kge/tree/MKBE) branch.

Two experiments are performed. In the first experiment the mKGE models are trained from scratch. The corresponding config files for the first experiment can be found [here](https://github.com/nluedema/kge/tree/master/experiments/search_config). In the second experiment the mKGE models are trained starting from pretrained structural embeddings.
The corresponding config files for the second experiment can be found [here](https://github.com/nluedema/kge/tree/master/experiments/search_config_pretrained).

The results of all my experiments can be found [here](https://github.com/nluedema/kge/tree/master/experiments/results)

## Setup
```sh
git clone https://github.com/nluedema/kge.git
cd kge
pip install -e .
python -m nltk.downloader stopwords

cd data
sh download_mkge.sh
cd ..
```

## First experiment
### ComplEx, DKRL and LiteralE
* The multimodal data for FB15K-237 is stored [here](https://github.com/nluedema/kge/tree/master/experiments/fb15k-237/preprocessed_files) and for YAGO3-10 [here](https://github.com/nluedema/kge/tree/master/experiments/yago3-10/preprocessed_files)
* The search configs for ComplEx, DKRL and LiteralE are stored [here](https://github.com/nluedema/kge/tree/master/experiments/search_config)
  * Adjust `text.filename` and `numeric.filename` in the config files if available
* Run the searches

Example: DKRL on FB15K-237 using text and numeric information
```sh
kge start [path-to-repo]/experiments/search_config/fb15k-237/fb15k-237-literale-text-numeric.yaml --search.device_pool cuda:0 --search.num_workers 1
```
* Use [create_best_models_search_files.py](https://github.com/nluedema/kge/blob/master/experiments/scripts/create_best_models_search_files.py) to create configs that train the best models 5 times

Example: Create best model search folders for all FB15K-237 searches
```sh
cd [path-to-repo]/local/experiments
python ../../experiments/scripts/create_best_models_search_files.py --prefix *-fb15k-237-*
```
* Navigate to the best model search folders and use ```kge resume``` to train best models 5 times

Example: Train a best model configuration 5 times
```sh
kge resume . --search.device_pool cuda:0 --search.num_workers 1
```

### MKBE
* Switch to the MKBE branch `git checkout MKBE`
* Adjust the filepaths for the text and numeric data in [preprocess_fb15k-237-mkbe.py](https://github.com/nluedema/kge/blob/MKBE/data/preprocess/preprocess_fb15k-237-mkbe.py) and [preprocess_yago3-10-mkbe.py](https://github.com/nluedema/kge/blob/MKBE/data/preprocess/preprocess_yago3-10-mkbe.py)
  * The multimodal data in the MKBE branch is stored [here](https://github.com/nluedema/kge/tree/MKBE/experiments/fb15k-237/preprocessed_files) and [here](https://github.com/nluedema/kge/tree/MKBE/experiments/yago3-10/preprocessed_files)
* Create the modified datasets for MKBE as shown below
```sh
cd [path-to-repo]/data
rm -r fb15k-237
rm -r yago3-10
sh download_mkge.sh
cp -r fb15k-237 fb15k-237-text
cp -r fb15k-237 fb15k-237-numeric
cp -r fb15k-237 fb15k-237-text-numeric
cp -r yago3-10 yago3-10-text
cp -r yago3-10 yago3-10-numeric
cp -r yago3-10 yago3-10-text-numeric
python preprocess/preprocess_fb15k-237-mkbe.py --modality text fb15k-237-text
python preprocess/preprocess_fb15k-237-mkbe.py --modality numeric fb15k-237-numeric
python preprocess/preprocess_fb15k-237-mkbe.py --modality all fb15k-237-text-numeric
python preprocess/preprocess_yago3-10-mkbe.py --modality text yago3-10-text
python preprocess/preprocess_yago3-10-mkbe.py --modality numeric yago3-10-numeric
python preprocess/preprocess_yago3-10-mkbe.py --modality all yago3-10-text-numeric
```
* Run the searches for MKBE

Example: MKBE on YAGO3-10 using text and numeric information
```sh
kge start [path-to-repo]/experiments/search_config/yago3-10/yago3-10-mkbe-text-numeric.yaml --search.device_pool cuda:0 --search.num_workers 1
```

* To train the best models 5 times switch back to master
  * Create best model search folders as before
  * Switch back to MKBE and run the searches

## Second Experiment
### ComplEx, DKRL and LiteralE
* Switch to the master branch `git checkout master`
* Get the pretrained models as shown below
```sh
cd [path-to-repo]/experiments
mkdir pretrained
cd pretrained
wget http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/fb15k-237-complex.pt
wget http://web.informatik.uni-mannheim.de/pi1/libkge-models/yago3-10-complex.pt
```
* The search configs of the second experiment for ComplEx, DKRL and LiteralE are stored [here](https://github.com/nluedema/kge/tree/master/experiments/search_config_pretrained)
  * Adjust `text.filename` and `numeric.filename` as before
  * Adjust `pretrain.model_filename`
* Run searches as before

### MKBE
* Switch to the MKBE branch `git checkout MKBE`
* The search configs are stored [here](https://github.com/nluedema/kge/tree/MKBE/experiments/search_config_pretrained)
  * Adjust `pretrain.model_filename`
* Run searches as before
