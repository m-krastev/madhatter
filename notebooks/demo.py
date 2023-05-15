# %%
import sys
import os.path
# Addition to path to unlock relative import to the madhatter package
sys.path.append(os.path.abspath(os.path.pardir))
import matplotlib.pyplot as plt

from madhatter.benchmark import CreativityBenchmark

# Read the text
with open('carroll-alice.txt') as f:
    text = f.read()

# Initialize the benchmark
bench = CreativityBenchmark(text, 'Alice in Wonderland')


# %%
# We have easy access methods for relevant segmentations of the text
bench.sents
bench.words
bench.tokenized_sents
bench.tagged_words
bench.tagged_sents


# %%
# We have easy access to things like frequency distributions over the whole book
bench.book_postag_counts()

# %%
# Similarly, we can make use of the metrics without having to create the benchmark object. If we choose to instead integrate with a NLP library like SpaCy.

from madhatter import metrics, utils
from nltk import word_tokenize

sent = "The quick brown fox jumped over the lazy dog."
metrics.concreteness(word_tokenize(sent), utils.get_concreteness_df()) # type: ignore

# %%
# Finally, we can put it all together by generating an overall Report object containing metrics for the whole text like so:

report = bench.report()

print(report)

# This object can later be used inside a machine learning pipeline to learn features about text to be used in classification and other tasks. See experiment.ipynb for examples.

# %%
# We also have access to a few different preset plotting functions

bench.plot_postag_distribution()
bench.plot_transition_matrix()

# %%
# We also have access to a variety of different metrics about the text:

conc = bench.concreteness_ratings()

# Shows all words along with their respective concreteness ratings
list(zip(bench.lemmas(), conc))

# %%
# We can also implement our own plots with the functions available to us

import matplotlib.pyplot as plt

plt.figure(figsize=(32,8))
plt.plot(conc)

# %%
# We can also showcase more advanced metrics utilizing LLMs:
# Note the spikes, those are moments in the context with high predictability. Predictability is a measure of the LLM's confidence in a given context.
# Low points signify low predictability -- that is, the expected word is not as predictable by the model, while high points mean that the model found less difficulty predicting the text.

plt.figure(figsize=(20,8))
plt.plot(report.predictability)

# %%
# Note the surprisal metric. It shows how similar or dissimilar potential contextual replacements are. It is strictly defined as the average of the top K likeliest replacements of the word in the given context. Higher scores mean that the word was expected and not too unusual. Lower scores mean that the word was "surprising" in this context, and suggested replacements had low or no similarity with the actual word being used.

plt.figure(figsize=(20,8))
plt.plot(report.surprisal)

# %%
%%timeit
list(i for i in range(10))

# %%
from madhatter.benchmark import BookReport

# an arbitary norm based on some observations for max possible values in data, can be improved
norm = BookReport(title='', nwords=20_000, mean_wl=6, mean_sl=300, mean_tokenspersent=40, prop_contentwords=0.10, mean_conc=5, mean_img=7, mean_freq=5, prop_pos=None, surprisal=None, predictability=None)

bench.plot_report(global_dist = norm,
    categories=["mean_wl", "mean_sl", "prop_contentwords", "mean_tokenspersent", "mean_conc", "mean_img", "mean_freq"],
    include_llm = False, print_time=False, include_pos=False
)

# %%
from madhatter.models import predict_tokens, default_model

model, tok = default_model()
res = predict_tokens("the quick brown fox jumped over the", "fox", model, tok, return_tokens=True)

# %%
import pandas as pd

print(pd.DataFrame(list(zip(res[0], res[1]))).to_latex(index=False)
)

# %%

import numpy as np

np.gradient(res[0])


