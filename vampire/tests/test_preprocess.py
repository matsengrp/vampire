import pandas as pd

import vampire.preprocess_adaptive as pre


def test_filters():
    unfiltered = pd.read_csv("vampire/data/adaptive-filter-test.csv")
    correct = pd.read_csv("vampire/data/adaptive-filter-test.correct.csv")
    filtered = pre.apply_all_filters(unfiltered)
    assert correct.equals(filtered)
