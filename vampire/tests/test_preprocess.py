import pandas as pd
import random

import vampire.preprocess_adaptive as pre


def test_filters():
    unfiltered = pd.read_csv("vampire/data/adaptive-filter-test.csv")
    correct = pd.read_csv("vampire/data/adaptive-filter-test.correct.csv")
    filtered = pre.apply_all_filters(unfiltered)
    assert correct.equals(filtered)


def test_dedup():
    original = pd.read_csv("vampire/data/protein-dedup-test.csv")
    correct = pd.read_csv("vampire/data/protein-dedup-test.correct.csv")
    random.seed(1)
    deduped = pre.dedup_on_vjcdr3(original)
    assert correct.equals(deduped)
