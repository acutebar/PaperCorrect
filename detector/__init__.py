import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
import math


from .edge_detector import get_disc_kernel
from .edge_detector import paper_clean_fast
from .edge_detector import display_lines
from .curve_trace import LineDetector
from .fnfit import Curve, bump
from .energy import surface_fit, total_energy, quadratic_fit
from .gradient_descent import generate_flat_cloud, generate_random_smooth_cloud, run_gradient_descent, run_multi_start_optimization
