import pandas as pd
import numpy as np

def get_lower_and_upper_bounds(series, multiplier = 1.5):
    q1_range = round((len(series.index) * .25),0).astype(int)
    q1 = series[:q1_range]

    return q1