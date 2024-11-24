import pandas as pd 
import statsmodels.api as sm
import numpy as np 


def targeting_step(dataset, propensity_scores, pred_outcomes, 
                   treatment_label='A', outcome_label='Y', show_summary=False):
    """
    Function to carry out the targeting step of TMLE
    """

    # Define our clever covariates
    H1W = dataset[treatment_label]/propensity_scores
    H0W = (1 - dataset[treatment_label])/(1 - propensity_scores)
    X = pd.DataFrame({'H0W': H0W, 'H1W': H1W})

    # convert predicted outcomes to logit scale 
    logit_QAW = np.log(pred_outcomes/(1-pred_outcomes))

    # fluctuation/substitution parameter 
    model_epsilon = sm.GLM(dataset[outcome_label], X, family=sm.families.Binomial(), offset=logit_QAW)
    model_epsilon = model_epsilon.fit()

    if (show_summary):
        print(model_epsilon.summary())

    return model_epsilon



def infl_curve(df, corrected_Q1, corrected_Q0, propensity_scores, 
                treatment_label='A', outcome_label='Y'):

    d1 = (df[treatment_label] * (df[outcome_label] - corrected_Q1)/propensity_scores) + \
                corrected_Q1 - corrected_Q1.mean()

    d0 = ((1 - df[treatment_label]) * (df[outcome_label] - corrected_Q0))/(1 - propensity_scores) + \
                corrected_Q0 - corrected_Q0.mean()

    infl_curve = d1 - d0

    return infl_curve



def conf_int(df, corrected_Q1, corrected_Q0, propensity_scores, 
                treatment_label='A', outcome_label='Y'):

    infl_curv_val = infl_curve(df, corrected_Q1, corrected_Q0, propensity_scores, treatment_label, outcome_label)

    std_err = np.sqrt(np.var(infl_curv_val)/(df[outcome_label].count()))

    tmle_val = corrected_Q1.mean() - corrected_Q0.mean()

    upper_conf = tmle_val + 1.96*std_err 
    lower_conf = tmle_val - 1.96*std_err 

    return lower_conf, upper_conf