# this contains functions for plotting various charts with seaborn and matplotlib
#for quick analysis

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import linregress
from sklearn.metrics import explained_variance_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import TweedieRegressor

# import prepare_regression as pr

#Bar Chart
def bar_chart(df, x_col, y_col, title):
    sns.barplot(data=df, x=x_col, y=y_col)
    plt.title(title)
    plt.show()



#Bar Chart with mean line
def mean_bar_plot(df, x, y, title='Bar Plot with Mean Line'):
    ax = sns.barplot(data = df, x=x, y=y)
    plt.title(title)
    
    # calculate the mean value
    mean = np.mean(df[y])
    
    # add a line for the mean value
    plt.axhline(mean, color='r', linestyle='dashed', linewidth=2)
    
    # add the mean value annotation 
    ax.text(0, mean + 0.01, 'Mean: {:.2f}'.format(mean), fontsize=12)
    plt.show()



#Scatter Plot
def scatter_plot(df, x, y, title):
    sns.scatterplot(data = df, x=x, y=y)
    plt.title(title)
    plt.show()


#Line Plot
def line_plot(df, x, y, title):
    sns.lineplot(data = df , x=x, y=y)
    plt.title(title)
    plt.show()


#Bar Chart with color
def bar_chart_with_color(df, x, y, color, title):
    sns.barplot(data = df , x=x, y=y, color= color)
    plt.title(title)
    plt.show()


#Histogram
def hist_plot(df, x, y, title):
    sns.histplot(data = df, x=x, y=y)
    plt.title(title)
    plt.show()


#Crosstab
def crosstab_plot(df, x, y, normalize=False, title='Crosstab Plot'):
    ct = pd.crosstab(df[x], df[y])
    if normalize:
        ct = ct.div(ct.sum(1), axis=0)
    sns.heatmap(ct, annot=True, cmap='Blues')
    plt.title(title)
    plt.show()



#Pearson's R (Continuous 1 vs Continuous 2), tests for correlation #linear
def pearson_r(x, y):
    """
    Calculates the Pearson correlation coefficient (r) between two lists of data using scipy.stats.pearsonr.
    """
    r, p = pearsonr(x, y)
    return r



#Spearman (Continuous vs Continuous 2), tests for correlation #non-linear


def spearman_rho(x, y):
    """
    Calculates the Spearman rank correlation coefficient (rho) between two lists of data using scipy.stats.spearmanr.
    """
    rho, p = spearmanr(x, y)
    return rho





# One Sample T Test
def one_sample_ttest(target_sample, overall_mean, alpha = 0.05):
    t, p = stats.ttest_1samp(target_sample, overall_mean)
    
    return t, p/2, alpha


# T-Test One Tailed (Continuous 1 vs Discrete)

'''
compares mean of continuous variables for the different groups in 
the discrete variable
'''

def t_test_one_tailed(data1, data2, alpha=0.05, alternative='greater'):
    t, p = stats.ttest_ind(data1, data2, equal_var=False)
    if alternative == 'greater':
        p = p/2
    elif alternative == 'less':
        p = 1 - p/2
    if p < alpha:
        print('Reject the null hypothesis')
    else:
        print('Fail to reject the null hypothesis')
    return t, p


# T-Test Two Tailed (Continuous 1 vs Discrete)

'''
compares mean of continuous variables for the different groups in 
the discrete variable
'''

def t_test_two_tailed(data1, data2, alpha=0.05):
    t, p = stats.ttest_ind(data1, data2, equal_var=False)
    if p < alpha:
        print('Reject the null hypothesis')
    else:
        print('Fail to reject the null hypothesis')
    return t, p



# Chi-Square (Discrete vs Discrete)
# testing dependence/relationship of 2 discrete variables

def chi_square_test(data1, data2, alpha=0.05):
    observed = pd.crosstab(data1, data2)
    chi2, p, degf, expected = stats.chi2_contingency(observed)

    if p < alpha:
        print('Reject the null hypothesis')
    else:
        print('Fail to reject the null hypothesis')
    



#ANOVA (Continuous 1 vs Discrete)
def anova_test(data, groups, alpha=0.05):
    f_val, p_val = stats.f_oneway(*data)
    if p_val < alpha:
        print('Reject the null hypothesis')
    else:
        print('Fail to reject the null hypothesis')
    return f_val, p_val



# Sum of Squared Errors SSE
def sse(y_true, y_pred):
    sse = mean_squared_error(y_true, y_pred) * len(y_true)
    return sse


# Mean Squared Error MSE
def mse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return mse


#Root Mean Squared Error RMSE
def rmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return rmse


# Explained Sum of Squares ESS
def ess(y_true, y_pred):
    mean_y = np.mean(y_true)
    ess = np.sum((y_pred - mean_y)**2)
    return ess


# Total Sum of Squares TSS
def total_sum_of_squares(arr):
    return np.sum(np.square(arr))


# R-Squared R2

def calculate_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)


# Linear Regressions
'''
Quickly calculate r value, p value, and standard error
'''
def linear_regression(x, y):
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return slope, intercept, r_value, p_value, std_err




# Explained Variance -or- RSquared
def explained_variance(y_true, y_pred):
    evs = explained_variance_score(y_true, y_pred)
    print("Explained Variance: ", evs)
    return evs





def X_train_y_train_split(df, target):
    X_train = df.drop(columns = target)
    y_train = df[target]
    return X_train, y_train



'''
#separating target variable
X_train, y_train = X_train_y_train_split(train_model)
X_validate, y_validate = X_train_y_train_split(validate_model)
X_test, y_test = X_train_y_train_split(test_model)
'''



def GLM(power, alpha):
    # create the model object
    glm = TweedieRegressor(power=power, alpha=alpha)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    glm.fit(X_train, y_traintarget)

    # predict train
    y_train['value_pred_lm'] = glm.predict(X_train)

    # evaluate: rmse
    rmse_train = rmse(y_traintarget, y_train.value_pred_mean)

    # predict validate
    y_validate['value_pred_lm'] = glm.predict(X_validate)

    # evaluate: rmse
    rmse_validate = rmse(y_validatetarget, y_validate.value_pred_median)

    return print("RMSE for GLM using TweedieRegressor\nTraining/In-Sample: ", round(rmse_train), 
      "\nValidation/Out-of-Sample: ", round(rmse_validate))


    