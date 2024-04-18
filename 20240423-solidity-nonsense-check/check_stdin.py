#!/usr/bin/env python3
import argparse
import sys

from nonsense_scoring import dataset_from_pickle, tfidf_score_function


def get_scoring_function():
    """Nonsense scoring function, higher means more likely to be nonsense.
    Default thresholds:
    min_length = 6
    min_score = 8.2
    score_len_threshold=25
    score_len_penalty_exp=0.9233
    score_rep_penalty_exp=0.9674
    """
    # Get the scoring function from nostril, and the n-gram data
    # import nostril
    # from nostril.nonsense_detector import _tfidf_score_function, dataset_from_pickle
    #
    # p = Path(nostril.__file__).parent / "ngram_data.pklz"
    # ngram_freq = dataset_from_pickle(str(p))
    # return _tfidf_score_function(ngram_freq)

    ngram_freq = dataset_from_pickle()
    return tfidf_score_function(ngram_freq)


def main():
    f_score = get_scoring_function()

    buff = []
    inputs = [line.strip() for line in sys.stdin]

    for s in inputs:
        # skips anything with less than 6 letters
        if len(s) < 6:
            continue

        buff.append((f_score(s), s))

    buff = sorted(buff, key=lambda x: x[0])[::-1]
    print("\n".join(f"{a:>4.1f} {b}" for a, b in buff))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Score each line from stdin for likelyhood of being nonsense.\n"
            "Example usage, with solidity compiler and jq:\n"
            "solc --ast-compact-json CONTRACT.sol | tail -n +5 | jq -r '.nodes[] | select(.nodes) | .nodes[] | select(.kind == 'function') | .name' | python check.py"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    main()
