# %%
import sys
import os.path
# Addition to path to unlock relative import to the madhatter package
sys.path.append(os.path.abspath(os.path.pardir))

from madhatter import *
from madhatter.loaders import *

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from multiprocess.pool import Pool
# A progress bar to try to give an overall idea of the progress made.
from tqdm import tqdm
import pickle
from pathlib import Path

from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from IPython.display import display

input_length = 100_000

print_latex = False


# %%
# nlp = spacy.load("en_core_web_sm", disable=[
#                  "ner",
#                  #  "lemmatizer",
#                  "textcat", "attribute_ruler"])
# nlp.pipe_names


def read_texts(path: Path | str, length: int = 1_000_000) -> list[str]:
    """Returns a list of strings sequentially read from the path specified as the option. 

    Parameters
    ----------
    path : Path
        Path to read from. The document will be opened in text-mode.
    length : int, optional
        The desired length of all texts, by default 1_000_000

    Returns
    -------
    list[str]
        List of the read character sequences.
    """
    with open(path) as f:
        text = []
        line = f.read(length)
        while len(line) > 0:
            text.append(line)
            line = f.read(length)
    return text


def split_strings(string: str, length=1_000_000):
    ret = []
    i = 0
    read = string[i*length:(i+1)*length]
    while len(read) > 0:
        ret.append(read)
        i += 1
        read = string[i*length:(i+1)*length]
    return ret


# %% [markdown]
# SpaCy performance concerns
# 
# | Processes | Total Time (s) | Peak Total Memory (MB) |
# | --- | --- | --- |
# | 1 (SpaCy pipe) | 25.104 | 6487 |
# | 16 (SpaCy pipe) | 45.345 | 6340 |
# | 16 (multiprocessing) | 8.313 | 6679 |
# 

# %% [markdown]
# ### Memory usage of Spacy vs Custom Package
# | Framework | peak memory | increment |
# |-----------|-------------|-----------|
# | Spacy | 5089.13 MiB |  4465.29 MiB |
# | Mad Hatter| 434.81 MiB  | 48.75 MiB |
# 
# Increment here is the more important number as it tells us how memory usage peaks when performing a given operation.
# 

# %% [markdown]
# ## Experimentation with pipelines
# Here we prepare a pipeline that will take the list of resources and return a list of `Report` objects. Those `Report` objects are then fed into a Pandas dataframe for further analysis. For better performance, we use the `multiprocessing` module to parallelize the pipeline, as each text is largely independent.
# 
# 
# Example listing of the pipeline:
# ```python
# def pipeline(resources: list[str]):
#     reports = []
#     for resource in resources:
#         report = Report(resource)
#         reports.append(report)
#     return reports
# ```

# %%
def process(file: str, title: str | None = None) -> BookReport: 
    try:
        return CreativityBenchmark(file, title if title is not None else "unknown").report(print_time=False, include_pos=True)
    except:
        return BookReport('')


def process_texts(args, processes: int = 16):
    """Note: args should be of the form (file, title if any)"""
    with Pool(processes) as p:
        return p.starmap(process, tqdm(args, total=len(args)))


def save_results(results, savepath):
    with open(savepath, 'wb') as file:
        pickle.dump(results, file)


def load_results(savepath):
    with open(savepath, 'rb') as file:
        return pickle.load(file)


# %% [markdown]
# ## Measuring the Gutenberg/Fiction dataset
# Note the lack of variety here. Gutenberg only has 18 works, but they lead to 2124 texts of length 100000. This may be a somewhat flawed methodology so I recommend exploring more fictional works.

# %%
savepath_creative = Path("./results/creative.parquet")

if savepath_creative.exists():
    creative_df = pd.read_parquet(savepath_creative)
else:
    from nltk.corpus import gutenberg

    creative_fns = [file for file in gutenberg.fileids()]
    creative_files = []
    for file in creative_fns:
        listt = split_strings(gutenberg.raw(creative_fns), length=input_length)
        creative_files.extend([(_, file) for _ in listt])

    print(len(creative_files))

    creative_results = process_texts(creative_files)
    
    creative_df = pd.DataFrame(creative_results)

    creative_df.insert(creative_df.shape[1], "class", "PG")

    creative_df.to_parquet(savepath_creative)

creative_df.hist(bins=30)
creative_df.head()


# %% [markdown]
# ## Loading legal datasets into the pipeline

# %%
legal_path = Path("./results/legal.parquet")
if legal_path.exists():
    legal_df = pd.read_parquet(legal_path)
else:
    from nltk.corpus import europarl_raw

    legal_texts = read_texts(ds_dgt(), length=input_length)

    europarl_txt = split_strings("".join([" ".join(
        [" ".join(para) for para in chap]) for chap in europarl_raw.english.chapters()]), length=input_length)
    legal_texts.extend(europarl_txt)

    legal_results = process_texts(
        [(legal_text, f"legal_text_{i}") for i, legal_text in enumerate(legal_texts)])

    legal_df = pd.DataFrame(legal_results)
    legal_df.insert(legal_df.shape[1], "class", "LG")

    legal_df.to_parquet(legal_path)


legal_df.hist(bins=30)
legal_df.head()


# %% [markdown]
# ## Loading writing prompts

# %%
wp_savepath = Path("./results/wp.parquet")
if wp_savepath.exists():
    writingprompts_df = pd.read_parquet(wp_savepath)
else:
    # TODO: Possibly try out stuff like actually splitting the writingprompts dataset instead of reading continuous text.

    wp_path = ds_writingprompts()
    writingprompts = read_texts(wp_path["train"][1], length=input_length)
    writingprompts.extend(read_texts(wp_path["test"][1], length=input_length))
    writingprompts.extend(read_texts(wp_path["val"][1], length=input_length))
    
    # Length (100_000 chars) = 100089
    print(f"Length of writingprompts dataset: {len(writingprompts)}")
    

    wp_results = process_texts(
        list((_, f"writingprompts_{i}") for i, _ in enumerate(writingprompts)))


    # Whole thing took around 35 minutes on battery charge
    
    writingprompts_df = pd.DataFrame(wp_results)
    writingprompts_df.insert(writingprompts_df.shape[1], "class", "WP")
    
    writingprompts_df.to_parquet(wp_savepath)


writingprompts_df.hist(bins=30)
writingprompts_df.head()


# %% [markdown]
# A little visualization of what is happening behind the scenes. It seems like the novels have quite a bit more variety behind them at first glance. 

# %% [markdown]
# ## Experiment
# 

# %% [markdown]
# After running the pipeline, we concatenate the results into a single dataframe which we can then use for further analysis. 

# %%
# Join
df = pd.concat([creative_df.head(2000), writingprompts_df.head(
    2000), legal_df.head(2000)], ignore_index=True)
df = df.join(pd.json_normalize(df["prop_pos"]).fillna(0.0))  # type: ignore

df = df.drop(columns=['predictability', 'surprisal'])


df["class"] = df["class"].astype('category')

def remove_outliers(df, deviation: float = 3) -> pd.DataFrame:
    # Remove outliers

    df = df.copy()

    cols = df.select_dtypes('number').columns
    df_sub = df.loc[:, cols]  # type: ignore
    lim = np.abs((df_sub - df_sub.mean()) / df_sub.std(ddof=0)) < deviation

    df.loc[:, cols] = df_sub.where(lim, np.nan)

    return df


df = remove_outliers(df, 3)

# fix bugged sentence length
df["mean_sl"] = df["mean_sl"].where(np.abs(
    (df["mean_sl"] - df["mean_sl"].mean()) / df["mean_sl"].std()) < 0.9, np.nan)

df = df.dropna()

# ####################

# drop unneeded columns and select features
xdf = df.drop(["title", "class", "prop_pos"], axis=1)

# drop arbitrary columns to see how results change
xdf = xdf.drop([
    # "nwords",
    # "mean_wl",
    # "mean_sl",

    # "NOUN",
    # "ADJ",
    # "VERB",
    # "mean_conc",
    # "mean_img",
    # "mean_freq",

    # "prop_contentwords",
    # "mean_tokenspersent"
], axis=1)
ydf = df["class"]

# make the splits
xtrain, xtest, ytrain, ytest = train_test_split(xdf, ydf, train_size=0.8)
xtest, xval, ytest, yval = train_test_split(xtest, ytest, test_size=0.5)

# create the pipeline
model = Pipeline(steps=[("scaler", StandardScaler()),
                 ("logistic", LogisticRegression(max_iter=200))])
split = PredefinedSplit([-1]*len(xtrain)+[0]*len(xval))
params = {'logistic__C': [1/64, 1/32, 1/16,
                          1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32, 64]}
# search = GridSearchCV(model, params, cv=split,
#                       n_jobs=None, verbose=False, refit=False)
search.fit(pd.concat([xtrain, xval]), pd.concat([ytrain, yval]))
model = model.set_params(**search.best_params_)
model.fit(xtrain, ytrain) 


ptrain = model.predict(xtrain)
pval = model.predict(xval)
ptest = model.predict(xtest)

experiment_dict = {
    'Experiment': 'Document Classification',
    'Size of Data': [
        len(xtrain),
        len(xval),
        len(xtest)
    ],
    'Accuracy': [
        accuracy_score(ytrain, ptrain),
        accuracy_score(yval, pval),
        accuracy_score(ytest, ptest)
    ],
    'Precision': [
        precision_score(ytrain, ptrain, average='macro'),
        precision_score(yval, pval, average='macro'),
        precision_score(ytest, ptest, average='macro')
    ],

    'Recall': [
        recall_score(ytrain, ptrain, average='macro'),
        recall_score(yval, pval, average='macro'),
        recall_score(ytest, ptest, average='macro')
    ],


    'F1-Score': [
        f1_score(ytrain, ptrain, average='macro'),
        f1_score(yval, pval, average='macro'),
        f1_score(ytest, ptest, average='macro')
    ],

}

experiment_df = pd.DataFrame(experiment_dict).T
experiment_df.columns = pd.MultiIndex.from_product(
    [['Split'], ['Train', 'Val', 'Test']])

experiment_df = experiment_df.drop(experiment_df.index[0])

display(experiment_df)

# if print_latex:
#     print(experiment_df.T.style
#           .format(precision=3)
#           .to_latex(hrules=True, position_float='centering',

#                     # type: ignore
#                     label=f'tab:{"_".join(experiment_dict["Experiment"].lower().split())}',
#                     caption=f'Performance results for {experiment_dict["Experiment"]}',
#                     position='htbp'))


# hmap_path = f'./plots/document_classification/heatmap.png'
# cmap_aid = plt.subplots(dpi=300)
# sns.heatmap(confusion_matrix(model.predict(xtest), ytest), ax=cmap_aid[1])

# cmap_aid[0].savefig(hmap_path, bbox_inches='tight')


# %%
features = df.columns[df.columns.str.contains(
    "title|prop_pos|class") != True].to_list()

g = sns.pairplot(df, hue='class')
g.savefig('./plots/document_classification/big_distplot.png')


# %%
nrows = 3
ncols = 4

fig, axs = plt.subplots(nrows, ncols, figsize=(16, 11), dpi=200)
j = 0
for feature, ax in zip(features, axs.flatten()):
    g = sns.kdeplot(df, x=feature, ax=ax, hue='class', legend=False, fill=True)

# g.legend()
fig.legend(['WP', 'PG', 'LG'], loc='center right', fontsize='large')
axs[-1, -1].axis('off')

fig.savefig('./plots/distplots_classification/data_dist.png')


# %%
display_df = pd.DataFrame(
    model.coef_, columns=xdf.columns)  # type: ignore

display_df['categories'] = ydf.cat.categories

display(display_df.T)

# sns.catplot(display_df.T, x='' kind='bar')
# sns.barplot(display_df)
# plt.barh(display_df.index, display_df)


# %% [markdown]
# - Write about lemmatization approaches
# - Possibly make a diagram for how the process goes

# %% [markdown]
# ## Authorship Identification

# %%
number_authors = 1000
max_works = 30

# flag to turn on and off if works are to be split into chunks (default behaviour already takes a single chunk of `length` tokens)
chunks = False
pg_authorship_id_path = Path(
    f'./results/pgauthorship_{number_authors}.parquet')

if pg_authorship_id_path.exists():
    pg_df = pd.read_parquet(pg_authorship_id_path)
else:


    def open_pg(id: str):
        with open(f"./gutenberg/data/text/{id}_text.txt") as f:
            return f.read()

    csv = "./gutenberg/metadata/metadata.csv"
    pg = pd.read_csv(csv)

    authors = pg.groupby(['author'], group_keys=True).count(
    ).sort_values(by=['id'], ascending=False)['id']
    authors = authors.loc[authors.index.str.contains(
        r"Various|Anonymous|Unknown") != True]
    print(f"Uniquely identified authors in Project Gutenberg: {len(authors)}\n" +
          f"Uniquely identified pieces of literature: {len(pg)}")

    texts = {}
    for author in authors.index[:number_authors]:
        texts[author] = []
        for i, book in enumerate(list(pg.loc[pg["author"] == author].itertuples())):
            if i > max_works:
                break
            texts[author].append(book.id)

    # for book in list(pg.itertuples())[:5]:
    #     print(book)
    filesnf = 0
    processing_set = []
    for author, collection in texts.items():
        for text in collection:
            try:
                if not chunks:
                    processing_set.append(
                        (open_pg(text)[:100_000], f"{text}_{author}"))
                else:
                    for i, t in enumerate(split_strings(open_pg(text), length=input_length)):
                        processing_set.append((t, f"{text}_{i}_{author}"))
            except FileNotFoundError:
                filesnf += 1

    print(f"Files not found: {filesnf}")
    print(f"Total files: {sum(len(i) for i in texts.values())}")

    results = process_texts(processing_set)

    pg_df = pd.DataFrame(results)
    pg_df.insert(pg_df.shape[-1], "class", [_[-1]
                 for _ in pg_df["title"].str.split('_')])
    pg_df = pg_df.join(pd.json_normalize(
        pg_df["prop_pos"]).fillna(0.0))  # type: ignore

    pg_df.to_parquet(pg_authorship_id_path)

pg_df


# %%
# Distribution:
csv = "./gutenberg/metadata/metadata.csv"
pg = pd.read_csv(csv)

authors = pg.groupby(['author'], group_keys=True).count(
).sort_values(by=['id'], ascending=False)['id']
authors = authors.loc[authors.index.str.contains(
    r"Various|Anonymous|Unknown") != True]
print(f"Uniquely identified authors in Project Gutenberg: {len(authors)}\n" +
        f"Uniquely identified pieces of literature: {len(pg)}")

plt.plot(authors[:1000])
plt.ylabel('# works')
plt.xlabel('author')
plt.xticks(range(0,1001,100),range(0,1001,100));
        

# %% [markdown]
# ### Pipeline

# %%
# Join
df = pd.read_parquet(pg_authorship_id_path)

# drop unneeded columns and select features
xdf = df.drop(["title", "class", "prop_pos"], axis=1)

# drop arbitrary columns to see how results change
xdf = xdf.drop([
    # "nwords",
    # "mean_wl",
    # "mean_sl",

    # "NOUN",
    # "ADJ",
    # "VERB",
    # "mean_conc", "mean_img", "mean_freq",

    # "prop_contentwords",
    # "mean_tokenspersent"
], axis=1)
ydf = df["class"]

# make the splits
xtrain, xtest, ytrain, ytest = train_test_split(xdf, ydf, train_size=0.8)
xtest, xval, ytest, yval = train_test_split(xtest, ytest, test_size=0.5)

# create the pipeline
model = Pipeline(steps=[("scaler", StandardScaler()),
                 ("logistic", LogisticRegression(max_iter=200))])
split = PredefinedSplit([-1]*len(xtrain)+[0]*len(xval))
params = {'logistic__C': [1/64, 1/32, 1/16,
                          1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32, 64]}
search = GridSearchCV(model, params, cv=split,
                      n_jobs=None, verbose=False, refit=False)
search.fit(pd.concat([xtrain, xval]), pd.concat([ytrain, yval]))
model = model.set_params(**search.best_params_)
model.fit(xtrain, ytrain)  # apply scaling on training data


ptrain = model.predict(xtrain)
pval = model.predict(xval)
ptest = model.predict(xtest)


experiment_dict = {
    'Experiment': f'Authorship Identification ({number_authors})',
    'Size of Data': [
        len(xtrain),
        len(xval),
        len(xtest)
    ],
    'Accuracy': [
        accuracy_score(ytrain, ptrain),
        accuracy_score(yval, pval),
        accuracy_score(ytest, ptest)
    ],
    'Precision': [
        precision_score(ytrain, ptrain, average='macro'),
        precision_score(yval, pval, average='macro'),
        precision_score(ytest, ptest, average='macro')
    ],

    'Recall': [
        recall_score(ytrain, ptrain, average='macro'),
        recall_score(yval, pval, average='macro'),
        recall_score(ytest, ptest, average='macro')
    ],


    'F1-Score': [
        f1_score(ytrain, ptrain, average='macro'),
        f1_score(yval, pval, average='macro'),
        f1_score(ytest, ptest, average='macro')
    ],

}

experiment_df = pd.DataFrame(experiment_dict).T
experiment_df.columns = pd.MultiIndex.from_product(
    [['Split'], ['Train', 'Val', 'Test']])

experiment_df = experiment_df.drop(experiment_df.index[0])

display(experiment_df)

if print_latex:
    print(experiment_df.T.style
          .format(precision=3)
          .to_latex(hrules=True, position_float='centering',

                    # type: ignore
                    label=f'tab:{"_".join(experiment_dict["Experiment"].lower().split())}',
                    caption=f'Performance results for {experiment_dict["Experiment"]}',
                    position='htbp'))

hmap_path = f'./plots/authorship_identification/aid_{number_authors}.png'
cmap_aid = plt.subplots(dpi=300)
sns.heatmap(confusion_matrix(model.predict(xtest), ytest), ax=cmap_aid[1])
cmap_aid[0].savefig(hmap_path, bbox_inches='tight')


# %%
# Join

from typing import Literal


dataset_type: Literal['-k40', ""] = "-k40"
model: Literal['xl-1542M', 'small-117M', 'large-762M','medium-345M'] = "xl-1542M"
nsamples = 40_000
mgtresultspath = Path(f'./results/mgt_results_{nsamples}_{model}{dataset_type}.parquet')


if mgtresultspath.exists():
    df_mgtresults = pd.read_parquet(mgtresultspath)
else:

    mgt_paths = load_machinetext()
    mgt = mgt_paths[model + dataset_type][0]
    non_mgt = mgt_paths["webtext"][0]

    with open(mgt) as f:
        mgt = pd.read_json(f, lines=True)

    with open(non_mgt) as f:
        non_mgt = pd.read_json(f, lines=True)

    mgt["class"] = "MGT"
    non_mgt["class"] = "HUMAN"
    df = pd.concat([mgt, non_mgt])
    df = df.reset_index()
    df["class"] = df["class"].astype('category')

    sample = pd.concat([df.loc[df["class"] =='MGT'].sample(nsamples//2), df.loc[df["class"] == 'HUMAN'].sample(nsamples//2)])

    results = process_texts(
        [(sample["text"][i], sample["class"][i]) for i in sample.index])

    df_mgtresults = pd.DataFrame(results)
    df_mgtresults["class"] = df_mgtresults["title"].astype('category')

    df_mgtresults.to_parquet(mgtresultspath)


# %%
# df = df.reset_index()
df = df_mgtresults

# drop unneeded columns and select features
xdf = df.drop(["title", "class", "prop_pos", "surprisal","predictability"], axis=1)

# drop arbitrary columns to see how results change
xdf = xdf.drop([
    # "nwords",
    # "mean_wl",
    # "mean_sl",

    # "NOUN",
    # "ADJ",
    # "VERB",
    # "mean_conc", "mean_img", "mean_freq",

    # "prop_contentwords",
    # "mean_tokenspersent"
], axis=1)
ydf = df["class"]

# make the splits
xtrain, xtest, ytrain, ytest = train_test_split(xdf, ydf, train_size=0.8, random_state=42)
xtest, xval, ytest, yval = train_test_split(xtest, ytest, test_size=0.5, random_state=42)

# create the pipeline
model = Pipeline(steps=[("scaler", StandardScaler()),
                 ("logistic", LogisticRegression(max_iter=200))])
split = PredefinedSplit([-1]*len(xtrain)+[0]*len(xval))
params = {'logistic__C': [1/64, 1/32, 1/16,
                          1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32, 64]}
search = GridSearchCV(model, params, cv=split,
                      n_jobs=None, verbose=False, refit=False)
search.fit(pd.concat([xtrain, xval]), pd.concat([ytrain, yval]))
model = model.set_params(**search.best_params_)
model.fit(xtrain, ytrain)  # apply scaling on training data

ptrain = model.predict(xtrain)
pval = model.predict(xval)
ptest = model.predict(xtest)

experiment_dict = {
    'Experiment': 'MGT Detection',
    'Size of Data': [
        len(xtrain),
        len(xval),
        len(xtest)
    ],
    'Accuracy': [
        accuracy_score(ytrain, ptrain),
        accuracy_score(yval, pval),
        accuracy_score(ytest, ptest)
    ],
    'Precision': [
        precision_score(ytrain, ptrain, average='macro'),
        precision_score(yval, pval, average='macro'),
        precision_score(ytest, ptest, average='macro')
    ],

    'Recall': [
        recall_score(ytrain, ptrain, average='macro'),
        recall_score(yval, pval, average='macro'),
        recall_score(ytest, ptest, average='macro')
    ],


    'F1-Score': [
        f1_score(ytrain, ptrain, average='macro'),
        f1_score(yval, pval, average='macro'),
        f1_score(ytest, ptest, average='macro')
    ],

}

experiment_df = pd.DataFrame(experiment_dict).T
experiment_df.columns = pd.MultiIndex.from_product(
    [['Split'], ['Train', 'Val', 'Test']])

experiment_df = experiment_df.drop(experiment_df.index[0])

display(experiment_df)

if print_latex:
    print(experiment_df.T.style
        .format(precision=3)
        .to_latex(hrules=True, position_float='centering',
                    
                    # type: ignore
                    label=f'tab:{"_".join(experiment_dict["Experiment"].lower().split())}',
                    caption=f'Performance results for {experiment_dict["Experiment"]}',
                    position='htbp'))

hmap_path = f'./plots/mgt_detection/cmatrix_xl.png'
cmap_aid = plt.subplots(dpi=300)
sns.heatmap(confusion_matrix(model.predict(xtest), ytest,
            labels=ydf.cat.categories.tolist()), ax=cmap_aid[1], annot=True, fmt="g", cbar=False)
cmap_aid[0].savefig(hmap_path, bbox_inches='tight')



