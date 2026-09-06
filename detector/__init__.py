import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
import math


from .edge_detector import get_disc_kernel
from .edge_detector import paper_clean_fast
from .edge_detector import display_lines
from .curve_trace import LineDetector
from .fnfit import Curve
from .fnfit import bump
from .fnfit import fn_fit
from .fnfit import curve_fit
from .energy import bump_2d, surface_fit, total_energy, evaluate_penalties, evaluate_complexity
from .gradient_descent import generate_flat_cloud, generate_random_smooth_cloud, run_gradient_descent, run_multi_start_optimization
