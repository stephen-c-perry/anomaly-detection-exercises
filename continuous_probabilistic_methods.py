import pandas as pd
import numpy as np



def get_lower_and_upper_bounds (series, multiplier=1.5):

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lb = q1 - (iqr * multiplier)
    ub = q3 + (iqr * multiplier)

    return lb, ub


def sigma_rule(df, col, sd):
    z_scores = (df[col] - df[col].mean()) / df[col].std()
    df['temp_zscores'] = z_scores
    df = df[df['temp_zscores'].abs() >=sd]
    return df
