import numpy as np
import numpy.typing as ntp
from typing import Any, Literal, Optional
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic as ic
import pandas as pd

from .models import Prediction
from .utils import get_freq_df, mean, stopwords


def imageability(data: str | list[str], imageability_df: pd.DataFrame) -> Optional[float] | list[Optional[float]]:
    """Returns the mean imageability rating for a given word or list of words, according to the table of ~40,000 words and word definitions, as defined by Brysbaert et al (2013)."""
    # TODO: Possibly look at amortized values given standard deviations

    # Fastest way for lookups so far.
    dictionary = dict(
        zip(imageability_df["item"], imageability_df["rating"]))

    return _ratings(data, dictionary)

def concreteness(data: str | list[str], concreteness_df: pd.DataFrame) -> float | None | list[float | None]:
    """Returns the mean concreteness rating for a given word or list of words, according to the table of ~40,000 words and word definitions, as defined by Brysbaert et al (2013)."""
    # TODO: Possibly look at amortized values given standard deviations

    # Fastest way for lookups so far.
    conc = dict(
        zip(concreteness_df["Word"], concreteness_df["Conc.M"]))

    return _ratings(data, conc)

def frequency_ratings(data):
    """Returns log10 frequency for lemmatized words"""
    df_dict = get_freq_df("dict")  # 6652 entries

    return _ratings(data, df_dict)  # type: ignore


def _ratings(data, func: dict):
    """j"""

    if isinstance(data, str):
        return func.get(data.lower(), None)
    if isinstance(data, list):
        # type: ignore
        return [func.get(w.lower(), None) for w in data if w not in stopwords]

    raise TypeError(
        f"Inappropriate argument type for `word`. Expected `list` or `str`, but got {type(data)}")



def _word2vec_similarity(first: str, second: str, word2vecmodel) -> float | None:
    try:
        return word2vecmodel.similarity(first.lower(), second.lower())
    except:
        return None


def word2vec_similarity(preds: list[Prediction], word2vecmodel) -> list[float]:
    # Return the mean similarity between tokens
    return [mean(list(x for x in list(_word2vec_similarity(pred.word, sug, word2vecmodel) for sug in pred.suggestions) if x)) for pred in preds]


def _lin_similarity(first: str, second: str, pos: Literal['n', 'a', 's', 'r', 'v'], ic_dict) -> float | None:
    """Generates similarity using Lin Similarity via the information content present in ic_dict. For words with multiple definitions, we skip disambiguation and select the most common definition.

    Parameters
    ----------
    first : str
        The first word
    second : str
        Second word
    pos : one of 'n', 'a', 's', 'r', 'v'
        The part of speech of the given words
    ic_dict : dict
        Information content dictionary, must contain both words.

    Returns
    -------
    float | None
        A float result if both words can be found in the ic_dict.
    """

    try:
        return wn.synset(f"{first.lower()}.{pos}.1").lin_similarity(wn.synset(f"{second.lower()}.{pos}.1"), ic_dict)
    except:
        return None


def wordnet_lin_similarity(preds: list[Prediction], ic_dict, tag_to_wn: dict):

    # Return the mean similarity between tokens
    return [mean(list(x for x in list(_lin_similarity(pred.word, sug, tag_to_wn[pred.original_tag], ic_dict) for sug in pred.suggestions) if x)) for pred in preds]



def _wup_similarity(first: str, second: str, pos: Literal['n', 'a', 's', 'r', 'v']) -> float | None:
    try:
        return wn.synset(f"{first.lower()}.{pos}.1").wup_similarity(wn.synset(f"{second.lower()}.{pos}.1"))
    except:
        return None



def wordnet_wup_similarity(preds: list[Prediction], tag_to_wn: dict):
    """Returns the mean Wu-Palmer path similarity for the tokens."""
    return [mean(list(x for x in list(_wup_similarity(pred.word, sug, tag_to_wn[pred.original_tag]) for sug in pred.suggestions) if x)) for pred in preds]


def predictability(preds: list[Prediction]) -> ntp.NDArray:
    """Returns an array calculating the predictability metric, defined as a mean over the gradient of each prediction.
    Parameters
    ----------
    preds : list[Prediction]
        A list of predictions to be used

    Returns
    -------
    NDArray
        A numpy array for the predictability values of each prediction.
    """
    return -np.gradient(np.array(list(pred.probs for pred in preds)), axis=1).mean(axis=1)


def surprisal(preds: list[Prediction], word2vec_model: Any | None = None) -> ntp.NDArray:

    if word2vec_model is None:
        
        import gensim
        from nltk.data import find

        # TODO: Implement non-hardcoded with NLTK loading.
        word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
            str(find('models/word2vec_sample/pruned.word2vec.txt')), binary=False)

    return np.array(word2vec_similarity(preds, word2vec_model))

