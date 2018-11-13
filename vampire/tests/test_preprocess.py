import random

import vampire.common as common
import vampire.preprocess_adaptive as pre


def test_filters():
    unfiltered = common.read_data_csv('adaptive-filter-test.csv')
    correct = common.read_data_csv('adaptive-filter-test.correct.csv')
    filtered = pre.apply_all_filters(unfiltered)
    assert correct.equals(filtered)


def test_dedup():
    original = common.read_data_csv('vjcdr3-dedup-test.csv')
    correct = common.read_data_csv('vjcdr3-dedup-test.correct.csv')
    random.seed(1)
    deduped = pre.dedup_on_vjcdr3(original)
    assert correct.equals(deduped)
