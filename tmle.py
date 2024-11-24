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