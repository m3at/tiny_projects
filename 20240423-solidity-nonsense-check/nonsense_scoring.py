"""Inference code copied and adapted from https://github.com/casics/nostril

Licence same as the original:
LGPL-2.1 license
"""

import gzip
import pickle
import string
import sys
from collections import defaultdict
from pathlib import Path
from urllib.request import urlretrieve

_delchars = str.maketrans("", "", string.punctuation + string.digits + " ")

_nonalpha = string.punctuation + string.whitespace + string.digits
_delete_nonalpha = str.maketrans("", "", _nonalpha)


def dataset_from_pickle():
    """Download the pre-trained n-gram from github"""

    save_path = Path("/tmp/ngram_data.pklz")
    if not save_path.exists():
        urlretrieve(
            "https://github.com/casics/nostril/raw/master/nostril/ngram_data.pklz",
            save_path,
        )

    # Dirty import injection used at unpickle time
    import ng

    sys.modules["ngrams"] = ng

    with gzip.open(save_path, "rb") as f:
        return pickle.load(f)


def sanitize_string(s) -> str:
    # Translate non-ASCII character codes.
    s = s.encode("ascii", errors="ignore").decode()
    # Lower-case the string & strip non-alpha.
    return s.lower().translate(_delete_nonalpha)


def ngrams(s, n):
    """Return all n-grams of length 'n' for the given string 's'."""
    return [s[i : i + n] for i in range(len(s) - n + 1)]


def _highest_total_frequency(ngram_freq):
    """Given a dictionary of n-gram score values for a corpus, returns the
    highest total frequency of any n-gram.
    """
    return max(ngram_freq[n].total_frequency for n in ngram_freq.keys())


def tfidf_score_function(
    ngram_freq, len_threshold=25, len_penalty_exp=1.365, repetition_penalty_exp=1.159
):
    """Generate a function (as a closure) that computes a score for a given
    string.  This needs to be called to create the function like this:
        score_string = _tfidf_score_function(...args...)
    The resulting scoring function can be called to score a string like this:
        score = score_string('yourstring')
    The formula implemented is as follows:

        S = a string to be scored (not given here, but to the function created)

        ngram_freq = table of NGramData named tuples
        ngram_length = the "n" in n-grams
        max_freq = max frequency of any n-gram
        num_ngrams = number of (any) n-grams of length n in S
        length_penalty = pow(max(0, num_ngrams - len_threshold), len_penalty_exp)
        ngram_score_sum = 0
        for every n-gram in S:
            c = count of times the n-gram appears in S
            idf = IDF score of n-gram from ngram_freq
            tf = 0.5 + 0.5*( c/max_freq )
            repetition_penalty = pow(c, repetition_penalty_exp)
            ngram_score_sum += (tf * idf * repetition_penalty)
        final score = (ngram_score_sum + length_penalty)/(1 + num_ngrams)

    The repetition_penalty is designed to penalize strings that contain a lot
    of repeats of the same n-gram.  Such repetition is a strong indicator of
    junk strings like "foofoofoofoofoo".  It works on the principle that for
    an exponent value y between 1 and 2, c^y is equal to the value of c for c
    = 1, a little bit more than c for c = 2, a little bit more still than c
    for c = 3, and so on; in other words, progressively increases the value
    for higher counts.  We do this because we can't directly penalize strings
    on the basis of length (see below).

    The division by num_ngrams in the final step is a scaling factor to deal
    with different string lengths.  The need for a scaling factor comes from
    the fact that very long identifiers can be real, and thus length by
    itself is not a good predictor of junk strings.  Without a length scaling
    factor, longer strings would end up with higher scores simply because
    we're adding up n-gram score values.

    Though it's true that string length is not a predictor of junk strings,
    it is true that extremely long strings are less likely to be real
    identifiers.  The addition of length_penalty in the formula above is used
    to penalize very long strings.  Even though long identifiers can be real,
    there comes a point where increasing length is more indicative of random
    strings.  Exploratory analysis suggests that this comes around 50-60
    characters.  The formula is designed to add nothing until the length
    exceeds this, and then to progressively increase in value as the length
    increases.

    Finally, note the implementation uses the number of n-grams in the string
    rather than the length of the string directly.  The number of n-grams is
    proportional to the length of the string, but getting the size of a
    dictionary is faster than taking the length of a string -- this approach
    is just an optimization.
    """
    max_freq = _highest_total_frequency(ngram_freq)
    ngram_length = len(next(iter(ngram_freq.keys())))
    len_threshold = int(len_threshold)

    def score_function(s: str) -> float:
        s = sanitize_string(s)

        # We only score alpha characters.
        s = s.translate(_delchars)
        # Generate list of n-grams for the given string.
        string_ngrams = ngrams(s, ngram_length)
        # Count up occurrences of each n-gram in the string.
        ngram_counts = defaultdict(int)
        for ngram in string_ngrams:
            ngram_counts[ngram] += 1
        num_ngrams = len(string_ngrams)
        length_penalty = pow(max(0, num_ngrams - len_threshold), len_penalty_exp)
        score = (
            sum(
                ngram_freq[n].idf
                * pow(c, repetition_penalty_exp)
                * (0.5 + 0.5 * c / max_freq)
                for n, c in ngram_counts.items()
            )
            + length_penalty
        )
        return score / (1 + num_ngrams)

    return score_function
