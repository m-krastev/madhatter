# Mad Hatter

> A text analysis package for authors and data scientists alike.

Mad Hatter is a Python package that provides a variety of text analysis tools. It is designed to be used by both authors and data scientists alike. It is currently in development and is not yet ready for use. The package provides the following features for text analysis:

- Simple features (word count, sentence count, average tokens per sentence, average word length, average sentence length, etc.)
- Advanced psycholinguistically-motivated measures (concreteness, imageability, rare word usage, etc.)
- Context-dependent LLM-based features (surprisal, predictability, etc.)
  
... All optimized for data analysis, graphing, and easy integration with existing tools!

## Installation

Run the following command to install the package and its dependencies:

```bash
pip install madhatter
```

### Using NLTK features

We highly recommend also running NLTK's downloader module in order to have access to all of the features that Mad Hatter provides. To do so, simply run the following command:

```bash
python -m nltk.downloader all
```

## Usage

The package provides high-level abstractions for text analysis that can be used with any text. The following example shows how to use the package to analyze a simple text file:

```python
from madhatter.benchmark import CreativityBenchmark

text = "The quick brown fox jumped over the lazy dog."
bench = CreativityBenchmark(text)

bench.report()
>>> BookReport(title='unknown', nwords=10, mean_wl=3.7, mean_sl=45.0, mean_tokenspersent=10.0, prop_contentwords=0.1, mean_conc=4.0633333333333335, mean_img=5.359999999999999, mean_freq=-1.6792249660842167, prop_pos={'ADJ': 0.2, 'NOUN': 0.3, 'VERB': 0.1}, surprisal=None, predictability=None)
```

### Command Line Interface
Mad Hatter is also available as a CLI tool. Simply provide a filename to the CLI and it will generate a report for you. The following example shows how to use the CLI to generate a report for a text file:



```bash
> python -m madhatter -h
usage: madhatter [-h] [-p] [-u] [-m MAXTOKENS] [-c CONTEXT] [-t TITLE] [-d TAGSET] filename

A command-line utility for generating book project reports.

positional arguments:
  filename              text file to parse

options:
  -h, --help            show this help message and exit
  -p, --postag          whether to return a POS tag distribution over the whole text
  -u, --usellm          whether to run GPU-intensive LLMs for additional characteristics
  -m MAXTOKENS, --maxtokens MAXTOKENS
                        maximum number of predicted tokens for the heavyweight metrics. Tokens start from the beginning of text, -1 to read until the
                        end
  -c CONTEXT, --context CONTEXT
                        context length for sliding window predictions as part of heavyweight metrics
  -t TITLE, --title TITLE
                        optional title to use for the report project.
  -d TAGSET, --tagset TAGSET
                        tagset to use
```


### Advanced Usage

You may also choose to use the package's lower-level functions to create your own custom analysis pipeline or integrate with NLP packages such as [SpaCy](https://github.com/explosion/spaCy).

```python
from madhatter import metrics
from madhatter import benchmark

text = "The quick brown fox jumped over the lazy dog."
bench = benchmark.CreativityBenchmark(text)

bench.words
>>> ['The', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog', '.']

metrics.imageability(bench.words)
>>> [1.41, 2.45, 3.14, 4.2, 3.4, 3.65, 1.41, 2.42, 4.1, 0.0]
```

Of course, feel free to also contribute to the package's development by opening an issue or submitting a pull request!

## License

The project is released under the MIT license. See `LICENSE` for more information.