from typing import Any, Literal, NamedTuple
from nltk.data import find as find
from transformers import AutoTokenizer as AutoTokenizer, BertForMaskedLM as BertForMaskedLM
from typing import Any, Literal, overload

def sent_predictions(sent: str | list[str], bench: Any, model: Any, tokenizer: Any, return_tokens: bool = True, k: int = ...) -> list[Prediction]: ...

def predict_tokens(sent: str, masked_word: str, model, tokenizer, return_tokens: bool = True, k: int = ...) -> tuple[list[float], list[str]]: ...


class Prediction(NamedTuple):
    word: str
    original_tag: str
    suggestions: list[str]
    probs: list[float]
    def __bool__(self) -> bool: ...



def sliding_window_preds_tagged(words: list[tuple[str, str]], model: Any, tokenizer: Any, return_tokens: Literal[True, False] = False, k: int = 20, max_preds: int = 10, stopwords: set | None = None, tags_of_interest: set | None = None) -> list[Prediction]: ...


def sliding_window_preds(sent: str | list[str], model: Any, tokenizer: Any, return_tokens: bool = ...,
                         k: int = ..., n: int = ..., tags_of_interest: list | None = ..., stopwords=...): ...

def default_model(model_name: str = ...): ...

def default_word2vec(data) -> Any: ...