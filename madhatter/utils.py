import pkgutil
from io import BytesIO
from typing import Literal, Sequence

import numpy as np
import numpy.typing as ntp
import pandas as pd

# from .loaders import load_ concreteness, load_imageability


def mean(items: Sequence) -> float:
    return 0 if len(items) == 0 else sum(items)/len(items)


def cross_softmax(one: ntp.NDArray, two: ntp.NDArray, temp1=0.5, temp2=0.5):
    # return (torch.softmax(torch.from_numpy(results[1][:,0]), dim=0) @ torch.softmax(torch.from_numpy(results[1][:,1]), dim=0)).item()
    exps = np.exp(one*temp1)
    exps /= exps.sum()
    exps2 = np.exp(two*temp2)
    exps2 /= exps2.sum()
    return exps @ exps2


def slope_coefficient(one: ntp.NDArray, two: ntp.NDArray) -> float:
    """Returns the coefficient of the slope"""
    # Using the integrated function
    # return np.tanh(np.polyfit(x,y,1)[0])
    # Manually implementing slope equation
    return ((one*two).mean(axis=0) - one.mean()*two.mean(axis=0)) / ((one**2).mean() - (one.mean())**2)


def get_concreteness_df(_format: Literal['df', 'dict'] = "df") -> pd.DataFrame | dict:
    # load_concreteness()
    dataframe = pd.read_csv(BytesIO(pkgutil.get_data(
        __package__, 'static/concreteness/concreteness.csv')), sep="\t")  # type: ignore
    # concreteness_df = concreteness_df.set_index("Word").sort_index()

    return dataframe if _format == "df" else dict(
        zip(dataframe["Word"], dataframe["Conc.M"]))


def get_imageability_df(_format="df") -> pd.DataFrame | dict:
    """Returns a table of the imageability of ~40,000 words and word definitions, as defined by Brysbaert et al (2013)."""

    # load_imageability()
    # Dicts are the fastest way to make string accesses
    dataframe = pd.read_csv(BytesIO(pkgutil.get_data(
        __package__, "static/imageability/cortese2004norms.csv")), header=9)  # type: ignore
    return dataframe if _format == "df" else dict(
        zip(dataframe["item"], dataframe["rating"]))


def get_freq_df(_format) -> pd.DataFrame | dict:
    '''
    Key:

    Word = Word type (headword followed by any variant forms) - see pp.4-5

    PoS  = Part of speech (grammatical word class - see pp. 12-13)

    Freq = Rounded frequency per million word tokens (down to a minimum of 10 occurrences of a lemma per million)- see pp. 5

    Ra   = Range: number of sectors of the corpus (out of a maximum of 100) in which the word occurs

    Disp = Dispersion value (Juilland's D) from a minimum of 0.00 to a maximum of 1.00.
    '''

    df_freq = pd.read_csv(BytesIO(pkgutil.get_data(
        __package__, "static/frequency/frequency.csv")), encoding='unicode_escape', sep="\t")  # type: ignore

    # drop unnamed column one cuz trash source,
    # also remove the column storing word variants
    df_freq = df_freq.drop([df_freq.columns[0], df_freq.columns[3]], axis=1)

    # clean the data
    df_freq = df_freq.convert_dtypes()
    df_freq["Freq"] = pd.to_numeric(df_freq["Freq"], errors="coerce")
    df_freq["Ra"] = pd.to_numeric(df_freq["Ra"], errors="coerce")
    df_freq["Disp"] = pd.to_numeric(df_freq["Disp"], errors="coerce")
    df_freq = df_freq.dropna()

    # filter out the word variants
    df_freq = df_freq.loc[df_freq["Word"] != '@']

    # replace the PoS tags with the ones we are using
    def replace_df(_df, column, replacements) -> pd.DataFrame:
        for src, tar in replacements:
            _df.loc[_df[column].str.contains(src), column] = tar
        return _df

    replacements = [("No", "NOUN"), ("Adv", "ADV"),
                    ("Adj", "ADJ"), ("Verb", "VERB")]
    df_freq = replace_df(df_freq, "PoS", replacements)
    
    # Scale logarithmically for a better result
    df_freq["Freq"] = np.log10(df_freq["Freq"])

    if _format == "df":
        return df_freq

    # # set the index to the "Word" column so lookups are faster
    # df = df.set_index('Word')

    # I don't particularly care enough for disambiguating their PoS tags :skull:, so might as well aggregate the columns and make it even faster.
    # group everything together because i literally cant bother with pos tag lookups on big scales
    df_freq = df_freq.groupby('Word').sum(numeric_only=True)

    # df_freq_dict = dict(zip(((x,y) for x, y in zip(df_freq.index, df_freq["PoS"]) if y in TAGS_OF_INTEREST), df_freq["Freq"])) # 5600 entries

    return dict(zip(df_freq.index, df_freq["Freq"]))  # 5900 entries


stopwords = {'of', 'been', "hadn't", "isn't", 'i', 'this', 'these', 'were', 'the', 'and', 'by', 'don', 'm', 'o', "wasn't", 'we', 'all', 'same', 'not', 'weren', 'at', 'those', 'few', 'shan', 'a', 'through', 'ain', 'its', 'how', "that'll", 'ours', 'you', 'here', 'nor', "weren't", 'myself', 'aren', 'why', "didn't", 'having', 'for', 'so', 'she', "mightn't", 'in', 'haven', 't', 'being', 'yourself', 'an', 'to', 'didn', 'between', 'them', "couldn't", "mustn't", 'itself', 'is', 'only', "aren't", 'very', "you'll", 'had', 'into', 'if', 'their', 'mustn', 'off', 'what', 'd', 'as', 'ourselves', 'that', 'hasn', 'each', 'me', 'below', "haven't", 'wouldn', 'shouldn', 'there', 'your', 'or', 'such', 'because', 'during', 'yourselves', 'other', 'hadn',
             "should've", 'own', 'mightn', 'our', 'y', 'after', 'on', "doesn't", 'ma', 'more', 'again', 'out', 'when', "you've", 'above', 'whom', 'under', 'have', 'll', 're', 've', 'isn', 'too', 'won', 'which', 'until', "you're", 'up', "hasn't", 'about', 'while', 'needn', 'wasn', 'doesn', 'once', 'he', 'my', 'they', 'him', 'does', 'her', 'most', 'am', 'further', 'then', 'some', 'herself', 'than', 'yours', 'over', 'down', 's', 'both', 'themselves', "won't", "shan't", 'can', "wouldn't", 'has', 'hers', 'did', 'against', 'be', "shouldn't", 'doing', "don't", 'will', 'his', 'no', 'should', "you'd", 'theirs', 'couldn', 'do', 'any', "it's", 'who', 'with', 'from', 'was', 'himself', 'it', 'just', 'now', 'but', 'before', "needn't", 'where', 'are', "she's"}.union("!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~")
