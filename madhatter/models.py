"""models.py
Base file for LLM operations on text.
"""

# Initialize models

import torch
from nltk import pos_tag, word_tokenize
from typing import Any, Literal, NamedTuple


class Prediction(NamedTuple):
    """Prediction class. Contains:
    ```
    word: str
    original_tag: str
    suggestions: list[str]
    probs: list[float]
    ```
    """
    word: str
    original_tag: str
    suggestions: list[str]
    probs: list[float]

    def __bool__(self):
        return len(self.suggestions) == len(self.probs)


def predict_tokens(sent: str, masked_word: str, model, tokenizer, return_tokens: bool = True, max_preds: int = 20) -> tuple[list[float], list[str]]:
    """
    Predict the top k tokens that could replace the masked word in the sentence. 

    Returns a list of tuples of the form (token, likelihood, similarity) where similarity is the cosine similarity of the given words in a word2vec model.

    Parameters
    ----------
    sent: str
        The sentence to predict tokens for.
    masked_word: str
        The word to predict tokens for. Note that this word must be in the sentence.
    model
        Must be a masked language model that takes in a sentence and returns a tensor of logits for each token
        in the sentence. Default assumes a pretrained BERT model from the HuggingFace `transformers` library.
    word2vec_model
        Must be a word2vec model that takes in a word and returns a vector representation of the word.
        Default is `gensim.models.keyedvectors.KeyedVectors` loaded from the `word2vec_sample` model 
        from the `nltk_data` package.
    k: int
        The number of tokens to return.    

    Returns
    -------
    List of tuples the form (token, likelihood)

    token: str
        The predicted token.
    likelihood: float
        The likelihood of the token being the masked word.
    """
    if masked_word not in sent:
        raise ValueError(f"{masked_word} not in {sent}")
    masked_sent = sent.replace(masked_word, "[MASK]")

    inputs = tokenizer(masked_sent, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    # retrieve index of [MASK]
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[
        0].nonzero(as_tuple=True)[0]

    vals, predicted_token_ids = torch.topk(  # pylint: disable=no-member
        logits[0, mask_token_index], max_preds, dim=-1)

    ret = []
    ret_tokens = []
    for i, predicted_token_id in enumerate(predicted_token_ids[0]):
        # if the actual tokens are needed, return those as well
        if return_tokens is True:
            word = tokenizer.decode(predicted_token_id)

            # If word is a subword, combine it with the previous word
            word = word if not word.startswith("##") else masked_word+word[2:]

            ret_tokens.append(word)
            
        ret.append(vals[0, i].item())

    return ret, ret_tokens


def sent_predictions(sent: str | list[str], model: Any, tokenizer: Any, return_tokens: Literal[True, False] = False, k: int = 20, stopwords: set | None = None, tags_of_interest: set | None = None) -> list[Prediction]:
    """Returns predictions for content words in a given sentence. If return_tokens is true, 
    returns a key-value pair dictionary where the key is the used word, and the value is a list of suggested tokens, 
    corresponding to the likekihoods in the first list.
    """
    if isinstance(sent, str):
        tokens = word_tokenize(sent.lower())
    elif isinstance(sent, list):
        tokens = [token.lower() for token in sent]
        sent = " ".join(tokens)
    else:
        raise TypeError()
    words = pos_tag(tokens, tagset='universal')

    results = []

    if stopwords is None:
        stopwords = set()

    if tags_of_interest is None:
        tags_of_interest = set()

    # loop over the words of the sentence
    for word, tag in words:
        # Early stopping
        if word in stopwords or tag not in tags_of_interest:
            continue
        
        probs, tokens = predict_tokens(
                sent, word, model, tokenizer, return_tokens=return_tokens, max_preds=k)

        results.append(Prediction(word, tag, tokens, probs))

    return results


def sliding_window_preds_tagged(words: list[tuple[str, str]], model: Any, tokenizer: Any, return_tokens: Literal[True, False] = False, k: int = 20, max_preds: int = 10, stopwords: set | None = None, tags_of_interest: set | None = None) -> list[Prediction]:
    """
        Note: must be used in conjunction with a list of tuples with already tagged words.
    """
    if not isinstance(words, list):
        raise ValueError(
            "Incorrect values passed for `words`, expected a list of tuples")

    if stopwords is None:
        stopwords = set()

    if tags_of_interest is None:
        tags_of_interest = set()

    results = []

    # loop over the words of the sentence
    for i, (word, tag) in enumerate(words[k:-k], start=k):
        # Early stopping
        if word in stopwords or tag not in tags_of_interest:
            continue

        sent_tuples = words[i-k:i+k]
        sent = " ".join(_[0] for _ in sent_tuples)

        probs, tokens = predict_tokens(
                sent, word, model, tokenizer, return_tokens=return_tokens, max_preds=k)

        results.append(Prediction(word, tag, tokens, probs))

    return results


def sliding_window_preds(_words: list[str], model: Any, tokenizer: Any, return_tokens: Literal[True, False] = False, k: int = 20, max_preds: int = 10, stopwords: set | None = None, tags_of_interest: set | None = None) -> list[Prediction]:
    """
        Returns a list of predictions given the sliding window for context on the model predictions.
    """
    if not isinstance(_words, list):
        raise ValueError(
            "Incorrect values passed for `words`, expected a list of strings")

    if len(_words) < k:
        raise ValueError(
            f'The given window ({_words=}) contains less tokens than the requested sliding window({k=}); ')

    words = pos_tag(_words, tagset='universal')

    if stopwords is None:
        stopwords = set()

    if tags_of_interest is None:
        tags_of_interest = set()

    results = []

    # loop over the words of the sentence
    for i, (word, tag) in enumerate(words[k:-k], start=k):
        # Early stopping
        if word in stopwords or tag not in tags_of_interest:
            continue

        sent_tuples = words[i-k:i+k]
        sent = " ".join(_[0] for _ in sent_tuples)

        probs, tokens = predict_tokens(
                sent, word, model, tokenizer, return_tokens=return_tokens, max_preds=k)

        results.append(Prediction(word, tag, tokens, probs))


    return results


def default_model(model_name="bert-base-uncased"):
    from transformers import AutoTokenizer, BertForMaskedLM

    return BertForMaskedLM.from_pretrained(model_name), AutoTokenizer.from_pretrained(model_name)


def default_word2vec():
    import gensim
    from nltk.data import find

    return gensim.models.KeyedVectors.load_word2vec_format(
        str(find('models/word2vec_sample/pruned.word2vec.txt')), binary=False)
