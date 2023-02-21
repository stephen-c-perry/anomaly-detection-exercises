import pandas as pd
import numpy as np


def get_quartiles(series):
    '''
    Takes in a series (df.column) and returns 4 quartiles of 
    that series
    '''
    q1_end = int(len(series) * .25)
    q2_end = int(len(series) * .50)
    q3_end = int(len(series) * .75)

    q1 = series[:q1_end]
    q2 = series[q1_end:q2_end]
    q3 = series[q2_end:q3_end]
    q4 = series[q3_end:]

    return q1, q2, q3, q4





def get_lower_and_upper_bounds(series, multiplier = 1.5):

    q1_end = int(len(series) * .25)
    q2_end = int(len(series) * .50)
    q3_end = int(len(series) * .75)


    q1 = series[:q1_end]
    q2 = series[q1_end:q2_end]
    q3 = series[q2_end:q3_end]
    q4 = series[q3_end:]

    IQR = series[q1_end:q3_end]


    return q1, q2, q3, q4