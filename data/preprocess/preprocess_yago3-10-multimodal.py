#!/usr/bin/env python
"""Preprocess a KGE dataset into a the format expected by libkge.

Call as `preprocess_base.py --folder <name>`. The original dataset should be stored in
subfolder `name` and have files "train.txt", "valid.txt", and "test.txt". Each file
contains one field_map triple per line, separated by tabs.

During preprocessing, each distinct entity name and each distinct relation name
is assigned an index (dense). The index-to-object mapping is stored in files
"entity_ids.del" and "relation_ids.del", resp. The triples (as indexes) are stored in
files "train.del", "valid.del", and "test.del". Additionally, the splits
"train_sample.del" (a random subset of train) and "valid_without_unseen.del" and
"test_without_unseen.del" are stored. The "test/valid_without_unseen.del" files are
subsets of "valid.del" and "test.del" resp. where all triples containing entities
or relations not existing in "train.del" have been filtered out.

Metadata information is stored in a file "dataset.yaml".

"""

import util
from typing import List
from os import path
from dataclasses import dataclass
from collections import defaultdict

#def preprocess_numeric(file, args):
#    triples_by_predicate = defaultdict(list)
#    with open(path.join(args.folder, file), "r") as f:
#        data = list(
#            map(lambda s: s.strip().split("\t"), f.readlines())
#        )
#        S, P, O = (0,1,2)
#
#        for t in data:
#            triples_by_predicate[t[P]].append([t[S],t[O]])
#    
#    for predicate, triples in triples_by_predicate.items():
#        with open(path.join(args.folder,f"{predicate}_preprocessed.txt"), "w") as f:
#            for t in triples:
#                f.write(f"{t[0]}\t{t[1]}\n")
#    
#    return triples_by_predicate.keys()

def preprocess_text(file, modality_name, args):
    with open(path.join(args.folder, file), "r") as f:
        data = list(
            map(lambda s: s.strip().split("\t"), f.readlines())
        )
    with open(path.join(args.folder, f"{modality_name}_preprocessed.txt"), "w") as f:
        for t in data:
            f.write(f"{t[0]}\t{modality_name}\t{t[1]}\n")

if __name__ == "__main__":
    args = util.default_parser().parse_args()
    field_map = {
        "S": args.subject_field,
        "P": args.predicate_field,
        "O": args.object_field,
    }

    print(f"Preprocessing {args.folder}...")

    # register raw splits
    train_raw = util.RawSplit(
        file="train.txt",
        field_map=field_map,
        collect_entities=True,
        collect_relations=True,
    )
    valid_raw = util.RawSplit(file="valid.txt", field_map=field_map,)
    test_raw = util.RawSplit(file="test.txt", field_map=field_map,)

    # create raw dataset
    raw_dataset = util.create_raw_dataset(
        train_raw, valid_raw, test_raw, args, create_splits=False)
    
    # add multimodal information
    text_modality_name = "hasDescription"
    preprocess_text("text_description.txt", text_modality_name , args)
    text_raw = util.RawMultimodalSplit(
        file=f"{text_modality_name}_preprocessed.txt",
        field_map = {
            "S": 0,
            "P": 1,
            "O": 2,
        },
        modality_name=text_modality_name
    )

    numeric_raw = util.RawMultimodalSplit(
        file="numerical_data_merged.txt",
        field_map = {
            "S": 0,
            "P": 1,
            "O": 2
        },
        modality_name="hasDate"
    )
    raw_multimodal_splits = [text_raw,numeric_raw]

    raw_dataset = util.add_multimodal_to_raw_dataset(
        raw_multimodal_splits,raw_dataset, folder=args.folder
    )

    # create splits: TRAIN
    train = util.Split(
        raw_split=train_raw,
        key="train",
        options={"type": "triples", "filename": "train.del", "split_type": "train"},
    )
    train_sample = util.SampledSplit(
        raw_split=train_raw,
        key="train_sample",
        sample_size=len(valid_raw.data),
        options={
            "type": "triples",
            "filename": "train_sample.del",
            "split_type": "train",
        },
    )
    train_raw.splits.extend([train, train_sample])

    # create splits: VALID
    valid = util.Split(
        raw_split=valid_raw,
        key="valid",
        options={"type": "triples", "filename": "valid.del", "split_type": "valid"},
    )
    valid_wo_unseen = util.FilteredSplit(
        raw_split=valid_raw,
        key="valid_without_unseen",
        filter_with=train_raw,
        options={
            "type": "triples",
            "filename": "valid_without_unseen.del",
            "split_type": "valid",
        },
    )
    valid_raw.splits.extend([valid, valid_wo_unseen])

    # create splits: TEST
    test = util.Split(
        raw_split=test_raw,
        key="test",
        options={"type": "triples", "filename": "test.del", "split_type": "test"},
    )
    test_wo_unseen = util.FilteredSplit(
        raw_split=test_raw,
        key="test_without_unseen",
        filter_with=train_raw,
        options={
            "type": "triples",
            "filename": "test_without_unseen.del",
            "split_type": "test",
        },
    )
    test_raw.splits.extend([test, test_wo_unseen])
    
    # do the work
    util.process_splits(raw_dataset)
    util.update_string_files(raw_dataset, args)
    util.write_dataset_yaml(raw_dataset.config, args.folder)
