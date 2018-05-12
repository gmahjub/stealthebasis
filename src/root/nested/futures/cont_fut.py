import datetime
import numpy as np
import pandas as pd
import quandl

class ContFut(object):
    """description of class"""
    def __init__(self, symbol):

        self.symbol = symbol

    def futures_rollover_weights(start_date, expiry_dates, contracts, rollover_days = 5):
        """ Construct a pandas dataframe that contains weights (between 0.0 and 1.0)
        of contract positions to hold in order to carry out a rollover of rollover_days
        prior to the expiration of the earliest contract. The matrix can then be "multiplied"
        with another DataFrame containing the settle prices of each contract in order to produce
        a continuous time series futures contract."""

        # construct a sequence of dates beginning from the earliest contract start
        # date to the end of the final contract
        dates = pd.date_range(start_date, expiry_dates[-1], freq='B')

        # create the 'roll weights' Dataframe that will store the mutlpliers for
        # each contract (between 0.0 and 1.0)
        roll_weights = pd.DataFrame(np.zeros((len(dates), len(contracts))), index = dates, columns = contracts)
        prev_date = roll_weights.index[0]

        # Loop through each contract and create the specific weightings for
        # each contract depending upon the settlement date and rollover_days
        for i, (item, ex_date) in enumerate(expiry_dates.iteritems()):
            if i < len(expiry_dates) - 1:
                roll_weights.ix[prev_date:ex_date - pd.offsets.BDay(), item] = 1
                roll_rng = pd.date_range(end=ex_date - pd.offsets.BDay(),
                                     periods=rollover_days + 1, freq='B')

                # Create a sequence of roll weights (i.e. [0.0,0.2,...,0.8,1.0]
                # and use these to adjust the weightings of each future
                decay_weights = np.linspace(0, 1, rollover_days + 1)
                roll_weights.ix[roll_rng, item] = 1 - decay_weights
                roll_weights.ix[roll_rng, expiry_dates.index[i+1]] = decay_weights
            else:
                roll_weights.ix[prev_date:, item] = 1
            prev_date = ex_date
        return roll_weights


    if __name__ == "__main__":
        # example WTI crude, download from Quandl
        wti_near = quandl.get("CME/CLF2014")
        wti_far = quandl.get("CME/CLG2014")
        wti = pd.DataFrame({"CLF2014": wti_near['Settle'],
                            "CLG2014": wti_far['Settle']}, index = wti_far.index)

        # create the dictionary of expiry dates for each contract
        expiry_dates = pd.Series({'CLF2014': datetime.datetime(2013,12,19),
                                  'CLG2014': datetime.datetime(2014, 2, 21)}).sort_values()

        # obtain the rollover weighting matrix/DataFrame
        weights = futures_rollover_weights(wti_near.index[0], expiry_dates, wti.columns)

        # construct the continuous futures of the WTI CL contracts
        wti_cont_fut = (wti * weights).sum(1).dropna()

        # output merged series of contract settle prices
        print(wti_cont_fut)
        #wti_cont_fut.tail(60)
