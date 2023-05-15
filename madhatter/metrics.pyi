
from typing import Any, Optional
from .utils import get_freq_df as get_freq_df, mean as mean, stopwords as stopwords
from .models import Prediction as Prediction
from typing import Any, Literal, Optional, overload

import numpy.typing as ntp
import pandas as pd

from .models import Prediction

@overload
def imageability(
    data: str, imageability_df: pd.DataFrame) -> Optional[float]: ...


@overload
def imageability(
    data: list[str], imageability_df: pd.DataFrame) -> list[Optional[float]]: ...


@overload
def _ratings(data: str, func: dict) -> float | None: ...


@overload
def _ratings(data: list[str], func: dict) -> list[float | None]: ...


@overload
def concreteness(data: str, func: dict) -> float | None: ...


@overload
def concreteness(data: list[str], func: dict) -> list[float | None]: ...


def _word2vec_similarity(first: str, second: str,
                         word2vecmodel) -> float | None: ...


def word2vec_similarity(
    preds: list[Prediction], word2vecmodel) -> list[float]: ...


def _lin_similarity(first: str, second: str,
                    pos: Literal['n', 'a', 's', 'r', 'v'], ic_dict) -> float | None: ...


def wordnet_lin_similarity(
    preds: list[Prediction], ic_dict, tag_to_wn: dict): ...
def _wup_similarity(first: str, second: str,
                    pos: Literal['n', 'a', 's', 'r', 'v']) -> float | None: ...


def wordnet_wup_similarity(preds: list[Prediction], tag_to_wn: dict): ...
def predictability(preds: list[Prediction]) -> ntp.NDArray: ...


def surprisal(preds: list[Prediction],
              word2vec_model: Any | None = None) -> ntp.NDArray: ...


def frequency_ratings(data): ...