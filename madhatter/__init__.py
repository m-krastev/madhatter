"""
    Mad Hatter.
"""

from .benchmark import *
from .metrics import *
from .models import *

import os.path

if not os.path.exists(f"{__file__}/static"):
    from .loaders import load_concreteness, load_freq, load_imageability

    load_freq()
    load_concreteness()
    load_imageability()
