"""Provides the data loaders for the datasets that may be used in the pipelines."""

import tarfile
import zipfile

from io import BytesIO
from pathlib import Path

from requests import get
from tqdm import tqdm


def ds_cloze(path="./data") -> dict[str, Path]:
    """Returns a dataset object for interacting with the cloze test dataset as extracted by XX

    Parameters
    ----------
    path : str, optional
        Default path for storing the files, by default "./data/"

    Returns
    -------
    dict[str,Path]
        A dictionary object with the following structure:
        ```
        - split: path
        ```
        Where source is one of `[test, train, val]` and path is a Path object pointing to the csv file of the dataset.

    Example Usage
    -------
    ```
    import pandas as pd

    ds = ds_cloze()
    df = pd.read_csv(ds["train"])
    df.head()
    ```
    """
    clozepath = Path(path) / "cloze/"

    trainpath = clozepath / "cloze_train.csv"
    testpath = clozepath / "cloze_test.csv"
    valpath = clozepath / "cloze_val.csv"
    if not clozepath.exists():
        clozepath.mkdir(exist_ok=True)
        trainpath.write_bytes(get("https://goo.gl/0OYkPK", timeout=5).content)

        testpath.write_bytes(get("https://goo.gl/BcTtB4", timeout=5).content)

        valpath.write_bytes(get("https://goo.gl/XWjas1", timeout=5).content)

    return {"test": testpath, "train": trainpath, "val": valpath}


def tiny_shakespeare(path="./data/"):
    """p """
    tiny_shakespeare_path = Path(
        path) / "tiny_shakespeare" / "tiny_shakespeare.txt"
    if not tiny_shakespeare_path.exists():
        tiny_shakespeare_path.write_bytes(get(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", timeout=5).content)

    return tiny_shakespeare_path


def ds_writingprompts(path="./data/") -> dict[str, tuple[Path, Path]]:
    """Returns a dataset object for interacting with the writing prompts dataset as extracted by Fan et al. (2015)

    Returns
    -------
    dict[str, tuple[Path, Path]]
        A dictionary object with the following structure:
        ```
        - split: (source, target)
        ```
        Where source is one of `[test, train, val]` and source and target are the "prompt" and "response(s)" files, respectively.

    Example Usage
    -------
    ```
    ds = ds_writingprompts()
    with open(ds["train"][1]) as f:
        f.read()
    ```
    """
    wppath = Path(path) / "writingPrompts/"
    if not wppath.exists():

        file = tarfile.open(fileobj=BytesIO(get(
            "https://dl.fbaipublicfiles.com/fairseq/data/writingPrompts.tar.gz", timeout=5).content))

        file.extractall(wppath.parent.parent)

    return {"test": (wppath / "test.wp_source", wppath / "test.wp_target"),
            "train": (wppath / "train.wp_source", wppath / "train.wp_target"),
            "val": (wppath / "valid.wp_source", wppath / "valid.wp_target")}


def ds_dgt(path="./data/") -> Path:
    """Returns the DGT-Acquis dataset offered by the European Union, etc.

    Parameters
    ----------
    path : str, optional
        Path to the data directory, by default "./data/"

    Returns
    -------
    Path
        A path reference for the available file that can be loaded next.
    """
    ds_path = Path(path) / "dgt" / "data.en.txt"
    if not ds_path.exists():
        with zipfile.ZipFile(BytesIO(get(
                "https://wt-public.emm4u.eu/Resources/DGT-Acquis-2012/data.en.txt.zip", timeout=5).content)) as file:
            file.extractall(ds_path.parent)

    return ds_path


def load_imageability() -> Path:
    """
    Loads the imageability dataset from Cortese et al. (2004) and returns the path to the file.
    """

    im_path = Path(__file__).parent / "static" / \
        "imageability" / "cortese20004norms.csv"

    if not im_path.exists():
        with zipfile.ZipFile(BytesIO(get(r'https://static-content.springer.com/esm/art%3A10.3758%2FBF03195585/MediaObjects/Cortese-BRM-2004.zip', timeout=5).content)) as file:
            file.extractall(im_path.parent)

        for file in im_path.parent.glob('**/*'):
            file.rename(im_path.parent / file.name)

        # (im_path.parent / "Cortese-BRMIC-2004").rmdir()
    return im_path

def load_concreteness() -> Path:
    conc_path = Path(__file__).parent / "static" / \
        "concreteness" / "concreteness.csv"

    if not conc_path.exists():
        conc_path.parent.mkdir(exist_ok=True, parents=True)
        conc_path.write_bytes(get(r'http://crr.ugent.be/papers/Concreteness_ratings_Brysbaert_et_al_BRM.txt', timeout=5).content)

    return conc_path

def load_freq() -> Path:
    """_summary_

    Returns
    -------
    Path
        _description_
    """
    freq_path = Path(__file__).parent / "static" / \
        "frequency" / "frequency.csv"

    if not freq_path.exists():
        freq_path.parent.mkdir(exist_ok=True, parents=True)
        freq_path.write_bytes(get(r'https://ucrel.lancs.ac.uk/bncfreq/lists/1_1_all_alpha.txt', timeout=5).content)

    return freq_path

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

def load_machinetext(_path: str = './data/') -> dict:
    # Adapted from https://github.com/openai/gpt-2-output-dataset/blob/master/download_dataset.py
    path = Path(_path) / 'machinetext'
    if not path.exists():
        path.mkdir()

    _ds = [
        'webtext',
        'small-117M',  'small-117M-k40',
        'medium-345M', 'medium-345M-k40',
        'large-762M',  'large-762M-k40',
        'xl-1542M',    'xl-1542M-k40',
    ]

    _splits = [
        'train', 'valid', 'test'
    ]

    filenames = [f"{ds}.{split}.jsonl" for split in _splits for ds in _ds]

    # check if dir is empty
    if next(path.iterdir(), None) is None:

        for filename in filenames:
            r = get("https://openaipublic.azureedge.net/gpt-2/output-dataset/v1/" + filename, timeout=30, stream=True)

            with open(path / filename, 'wb') as f:
                file_size = int(r.headers["content-length"])
                chunk_size = 1000
                with tqdm(ncols=100, desc="Fetching " + filename, total=file_size, unit_scale=True) as pbar:
                    # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        pbar.update(chunk_size)
    
    return {
        ds: [path / f"{ds}.{split}.jsonl" for split in _splits] for ds in _ds
    }


if __name__ == "__main__":
    print("You should use the functions defined in the file, not run it directly!")
