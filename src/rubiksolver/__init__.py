from .end_to_end import main
from .ManualSquareExtractor import ManualSquareExtractor
from .SolutionGallery import SolutionGallery
from .Solver import Solver
from .Cube import Cube
from .Cube import plt
import os
import sys

# Read the install dir to know where training data is
from .config import install_dir
sys.path.append(install_dir)
