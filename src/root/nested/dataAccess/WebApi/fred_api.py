from root.nested.SysOs.os_mux import OSMuxImpl
from root.nested.dataAccess.secure_keys_access import SecureKeysAccess
from root.nested.dataAccess.quandl_data_object import QuandlDataObject
from root.nested.visualize.extend_bokeh import ExtendBokeh
from root.nested import get_logger
import seaborn as sns
sns.set(style='ticks')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import operator
from fredapi import Fred
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

class FredApi:

    def __init__(self):

        self.logger = get_logger()
        self.source = 'fred'
        self.api_key = SecureKeysAccess.get_vendor_api_key_static(vendor=str.upper(self.source))
        self.fred_pwd = OSMuxImpl.get_proper_path('/workspace/data/fred/')
        self.seaborn_plots_pwd = OSMuxImpl.get_proper_path('/workspace/data/seaborn/plots/')
        self.fred = Fred(api_key=self.api_key)

    def search_fred_by_category(self,
                                category_id,
                                limit=20,
                                order_by='popularity',
                                sort_order='desc'):

        # https: // fred.stlouisfed.org / categories / 32413
        # the above is the category for BAML Total Return Bond Index category
        df_category_series = self.fred.search_by_category(category_id,
                                                          limit=limit,
                                                          order_by=order_by,
                                                          sort_order=sort_order)
        return (df_category_series)

    def get_all_series_in_category(self,
                                   category_id,
                                   limit = 20, # set it obnoxiously high so default is no limit
                                   observation_start='2010-01-01',
                                   observation_end = pd.datetime.now().strftime('%Y-%m-%d')):

        df_category_series = self.search_fred_by_category(category_id=category_id,
                                                          limit=limit,
                                                          order_by='popularity',
                                                          sort_order='desc')
        return_series = df_category_series.id.apply(
            self.get_data, args = (observation_start, observation_end))
        transposed = return_series.T
        transposed.dropna(inplace=True)
        return transposed

    def category_series_to_csv(self,
                               category_id,
                               path_to_file):

        df_category_series = self.search_fred_by_category(category_id=category_id,
                                                          order_by='popularity',
                                                          sort_order='desc')
        df_category_series.to_csv(path_to_file)

    def get_multiple_categories_series(self,
                                       category_id_list,
                                       limit_list=[20,20], # obnoxiously high for default of no limit
                                       observation_start='2010-01-01',
                                       observation_end = pd.datetime.now().strftime('%Y-%m-%d')):

        cat_limit_list = [tupled for tupled in zip(category_id_list, limit_list)]
        df_category_series_list = [self.search_fred_by_category(category_id=cat_id, limit=limit,
                                                                order_by='popularity', sort_order='desc')
                                   for cat_id, limit in cat_limit_list]
        joined_dataframe = pd.DataFrame()
        for df_category_series in df_category_series_list:
            return_series = df_category_series.id.apply(self.get_data, args = (observation_start, observation_end))
            transposed = return_series.T
            transposed.dropna(inplace=True)
            if joined_dataframe.empty is True:
                joined_dataframe = transposed
            else:
                joined_dataframe = joined_dataframe.join(transposed, how='inner')
        return joined_dataframe

    def get_data(self,
                 series_id,
                 observation_start='2010-01-01',
                 observation_end=pd.datetime.now().strftime("%Y-%m-%d")):

        data = self.fred.get_series(series_id,
                                    observation_start=observation_start,
                                    observation_end=observation_end)
        series_meta_info = self.fred.get_series_info(series_id=series_id)
        data.name = series_meta_info['title']
        return data

    def correlation_analysis(self,
                             px_df,
                             corr_heatmap_save_filename='corr_heatmap.png',
                             pairplot_save_filename='pairplot.png'):

        rets_df = px_df.pct_change().dropna()
        corr_matrix = rets_df.corr()
        # use seaborn for heatmap of correlation
        heatmap_plot = sns.heatmap(data=corr_matrix)
        heatmap_plot.get_figure().savefig(self.seaborn_plots_pwd + corr_heatmap_save_filename)
        pairplot = sns.pairplot(data=rets_df)
        pairplot.fig.savefig(self.seaborn_plots_pwd + pairplot_save_filename)
        plt.show()

    def interest_rates_autocorrelation(self,
                                       ir_class='EURODOLLARS',
                                       contract='ED4_WHITE',
                                       observation_start='2014-06-01',
                                       observation_end=pd.datetime.now().strftime('%Y-%m-%d'),
                                       which_lag=1):

        # for example, if you wanted to look at the EURODOLLAR, you would do 100 - ED(px).
        qdo_eurodollar = QuandlDataObject(ir_class,
                                          contract,
                                          '.csv')
        ed_df = qdo_eurodollar.get_df()
        ed_df = ed_df[observation_start:observation_end]
        # if nothing traded on a certain day, just drop the row - e.g. Dec 5th, 2018 - market
        # closed for the funeral of Georgy Bush the first.
        ed_df=ed_df[ed_df.Volume > 0.0]
        # settle price is the 2 pm CST print.
        # last price is the 4 pm CST close print.
        # open price is the 5 pm CST open print.
        result_series = ed_df.apply(self.create_open_settle_last_ed_ts, axis=1)
        result_series=result_series.T.fillna(0).apply(lambda x: sum(x), axis=1)
        result_series.name = 'px'
        price_diff = result_series.diff().dropna()
        price_diff.name = 'px_one_lag_diff'
        #ed_df.Settle.diff().dropna().plot(title="ed_df")
        #plt.show()
        #price_diff.plot(title="price_diff")
        #plt.show()
        autocorr_one_lag = price_diff.autocorr(lag=which_lag)
        plot_acf(price_diff, lags=10)
        plt.show()
        self.logger.info("FredApi.interest_rates_autocorrelation(): %s lag autocorr is %4.6f",
                         str(which_lag), autocorr_one_lag)
        # the highest autocorrelations are for Open-settle pairs, Settle - Last pairs, and
        # last - open pairs. And the autocorrelaiton is negative, which means there is some
        # reversion.

    def rolling_eurodollar_session_corr_pos_ind(self,
                                                row,
                                                correl_filter):

        #bool_val = correl_filter[2](correl_filter[0][0](row['lagged_corr_series'], correl_filter[0][1]),
        #                            correl_filter[1][0](row['lagged_corr_series'], correl_filter[1][1]))
        if (np.isnan(row['os_sl_lagged_corr_series']) or np.isnan(row['os_snxto_lagged_corr_series'])):
            return (0, 0, 0)
        os_sl_a_cond = correl_filter['os_sl'][0][0](row['os_sl_lagged_corr_series'], correl_filter['os_sl'][0][1])
        os_sl_b_cond = correl_filter['os_sl'][1][0](row['os_sl_lagged_corr_series'], correl_filter['os_sl'][1][1])
        os_snxto_a_cond = correl_filter['os_snxto'][0][0](row['os_snxto_lagged_corr_series'], correl_filter['os_sl'][0][1])
        os_snxto_b_cond = correl_filter['os_snxto'][1][0](row['os_snxto_lagged_corr_series'], correl_filter['os_sl'][1][1])
        ol_lnxto_a_cond = correl_filter['ol_lnxto'][0][0](row['ol_lnxto_lagged_corr_series'], correl_filter['os_sl'][0][1])
        ol_lnxto_b_cond = correl_filter['ol_lnxto'][1][0](row['ol_lnxto_lagged_corr_series'], correl_filter['os_sl'][1][1])
        os_sl_pos_ind = 0
        os_snxto_pos_ind = 0
        ol_lnxto_pos_ind = 0
        # based on corr(settle-open, last-settle)
        if (os_sl_a_cond):
            os_sl_pos_ind = -1
        elif (os_sl_b_cond):
            os_sl_pos_ind = 1
        # based on corr(open-settle, settle-nextOpen)
        if (os_snxto_a_cond):
            os_snxto_pos_ind = -1
        elif (os_snxto_b_cond):
            os_snxto_pos_ind = 1
        # based on corr(open-last, last-nextOpen)
        if (ol_lnxto_a_cond):
            ol_lnxto_pos_ind = -1
        elif (ol_lnxto_b_cond):
            ol_lnxto_pos_ind = 1
        return (os_sl_pos_ind, os_snxto_pos_ind, ol_lnxto_pos_ind)

    def rolling_eurodollar_os_sl_corr(self,
                                      ir_class="EURODOLLARS",
                                      contract='ED4_WHITE',
                                      observation_start='2014-06-01',
                                      observation_end=pd.datetime.now().strftime('%Y-%m-%d'),
                                      which_lag=1,
                                      rolling_window_size=60,
                                      rolling_pnl_window_size=90,
                                      execution_slippage=-0.0025,
                                      min_input_vol=0.0,
                                      pos_correl_filter_val=0.2,
                                      neg_correl_filter_val=-0.2):

        # default window size is one week,there are two observations per day.
        qdo_eurodollar = QuandlDataObject(ir_class,
                                          contract,
                                          '.csv')
        ed_df = qdo_eurodollar.get_df()
        ed_df = ed_df[observation_start:observation_end]
        # if nothing traded on a certain day, just drop the row - e.g. Dec 5th, 2018 - market
        # closed for the funeral of George Bush the first.
        ed_df = ed_df[ed_df.Volume > 0.0]
        ed_df['OpenSettleDelta'] = ed_df.Settle - ed_df.Open
        ed_df['OpenLastDelta'] = ed_df.Last - ed_df.Open
        ed_df['SettleLastDelta'] = ed_df.Last - ed_df.Settle
        ed_df['SettleNextOpenDelta'] = ed_df.Open.shift(periods=-which_lag) - ed_df.Settle
        ed_df['LastNextOpenDelta'] = ed_df.Open.shift(periods=-which_lag) - ed_df.Last
        conditions = [(pd.to_numeric(ed_df.OpenSettleDelta.mul(1000000.0), downcast='integer') > int(min_input_vol*1000000.0) ), # one tick = 5000
                      (pd.to_numeric(ed_df.OpenSettleDelta.mul(1000000.0), downcast='integer') < int(-min_input_vol*1000000.0))]  # one tick = 5000
        ol_delta_conditions = [(pd.to_numeric(ed_df.OpenLastDelta.mul(1000000.0), downcast='integer') > int(min_input_vol*1000000.0) ), # one tick = 5000
                      (pd.to_numeric(ed_df.OpenLastDelta.mul(1000000.0), downcast='integer') < int(-min_input_vol*1000000.0))]  # one tick = 5000
        ## the below, 1.0 or -1.0 multiples, tells us whether we expect reversion in next period,
        ## or autocorrelation. (-1.0,1.0) = reversion, (1.0, -1.0) = autocorrelation
        choices_settle_last = [ed_df.SettleLastDelta.mul(1.0), ed_df.SettleLastDelta.mul(-1.0)]
        choices_settle_nextopen = [ed_df.SettleNextOpenDelta.mul(1.0), ed_df.SettleNextOpenDelta.mul(-1.0)]
        choices_last_nextopen = [ed_df.LastNextOpenDelta.mul(1.0), ed_df.LastNextOpenDelta.mul(-1.0)]
        ed_df['SettleLastTradeSelect'] = np.select(conditions, choices_settle_last, default=0.0)
        ed_df['SettleNextOpenTradeSelect'] = np.select(conditions, choices_settle_nextopen, default=0.0)
        ed_df['LastNextOpenTradeSelect'] = np.select(ol_delta_conditions, choices_last_nextopen, default=0.0)

        ed_df['os_sl_corr_series'] = ed_df.OpenSettleDelta.rolling(rolling_window_size).corr(ed_df.SettleLastDelta)
        ed_df['os_snxto_corr_series'] = ed_df.OpenSettleDelta.rolling(rolling_window_size).corr(ed_df.SettleNextOpenDelta)
        ed_df['ol_lnxto_corr_series'] = ed_df.OpenLastDelta.rolling(rolling_window_size).corr(ed_df.LastNextOpenDelta)

        ed_df['rolling_reversion_trade_pnl'] = ed_df.SettleLastTradeSelect.rolling(rolling_pnl_window_size).\
            sum().div(0.005)
        ed_df['fwd_looking_rolling_reversion_trade_pnl'] = ed_df.rolling_reversion_trade_pnl.\
            shift(-1*rolling_pnl_window_size+1)

        ed_df['rolling_reversion_settleNextOpen_trade_pnl'] = ed_df.SettleNextOpenTradeSelect.\
            rolling(rolling_pnl_window_size).sum().div(0.005)
        ed_df['fwd_looking_rolling_reversion_settleNextOpen_trade_pnl'] = ed_df.rolling_reversion_settleNextOpen_trade_pnl.\
            shift(-1*rolling_pnl_window_size+1)

        ed_df['rolling_reversion_lastNextOpen_trade_pnl'] = ed_df.LastNextOpenTradeSelect.\
            rolling(rolling_pnl_window_size).sum().div(0.005)
        ed_df['fwd_looking_rolling_reversion_lastNextOpen_trade_pnl'] = ed_df.rolling_reversion_lastNextOpen_trade_pnl.\
            shift(-1*rolling_pnl_window_size+1)

        ed_df['os_sl_lagged_corr_series'] = ed_df.os_sl_corr_series.shift(periods=1)
        ed_df['os_snxto_lagged_corr_series'] = ed_df.os_snxto_corr_series.shift(periods=1)
        ed_df['ol_lnxto_lagged_corr_series'] = ed_df.ol_lnxto_corr_series.shift(periods=1)
        # create filter conditions for correlation
        correl_filter = {}
        correl_filter['os_sl'] = [(operator.gt, pos_correl_filter_val), (operator.lt, neg_correl_filter_val), operator.or_]
        correl_filter['os_snxto'] = [(operator.gt, pos_correl_filter_val), (operator.lt, neg_correl_filter_val), operator.or_]
        correl_filter['ol_lnxto'] = [(operator.gt, pos_correl_filter_val), (operator.lt, neg_correl_filter_val), operator.or_]
        pos_ind_series = ed_df.apply(self.rolling_eurodollar_session_corr_pos_ind,
                                     args=(correl_filter,), axis=1)
        ed_df['os_sl_pos_ind'] = pos_ind_series.apply(lambda pos_ind_tuple: pos_ind_tuple[0])
        ed_df['os_snxto_pos_ind'] = pos_ind_series.apply(lambda pos_ind_tuple: pos_ind_tuple[1])
        ed_df['ol_lnxto_pos_ind'] = pos_ind_series.apply(lambda pos_ind_tuple: pos_ind_tuple[2])
        np_os_sl_pos_ind = ed_df.os_sl_pos_ind.values
        np_os_snxto_pos_ind = ed_df.os_snxto_pos_ind.values
        np_ol_lnxto_pos_ind = ed_df.ol_lnxto_pos_ind.values
        np_array_list = [np.repeat(pos_ind, np.min([rolling_pnl_window_size,len(np_os_sl_pos_ind)-item_idx]))
                         for item_idx, pos_ind in enumerate(np_os_sl_pos_ind) ]
        final_np_array_list = [np.append(np.append(np.repeat(0, np.min([item_idx, len(np_array_list)-len(npa)])),
                                                   np.array(npa)), np.repeat(0, np.max([len(np_array_list)-(item_idx+len(npa)),0])))
                               for item_idx, npa in enumerate(np_array_list)]
        np_os_snxto_array_list = [np.repeat(pos_ind, np.min([rolling_pnl_window_size,len(np_os_snxto_pos_ind)-item_idx]))
                                  for item_idx, pos_ind in enumerate(np_os_snxto_pos_ind) ]
        final_np_os_snxto_array_list = [np.append(np.append(np.repeat(0, np.min([item_idx, len(np_os_snxto_array_list)-len(npa)])),
                                                   np.array(npa)), np.repeat(0, np.max([len(np_os_snxto_array_list)-(item_idx+len(npa)),0])))
                                        for item_idx, npa in enumerate(np_os_snxto_array_list)]
        np_ol_lnxto_array_list = [np.repeat(pos_ind, np.min([rolling_pnl_window_size,len(np_ol_lnxto_pos_ind)-item_idx]))
                         for item_idx, pos_ind in enumerate(np_ol_lnxto_pos_ind) ]
        final_np_ol_lnxto_array_list = [np.append(np.append(np.repeat(0, np.min([item_idx, len(np_ol_lnxto_array_list)-len(npa)])),
                                                   np.array(npa)), np.repeat(0, np.max([len(np_ol_lnxto_array_list)-(item_idx+len(npa)),0])))
                               for item_idx, npa in enumerate(np_ol_lnxto_array_list)]
        self.logger.info("FredAPI:rolling_eurodollar_os_sl_corr(): final_np_array list dimensions are %s", np.array(final_np_array_list).shape)
        os_sl_total_pos_ind = np.sum(np.array(final_np_array_list), axis=0)
        os_snxto_total_pos_ind = np.sum(np.array(final_np_os_snxto_array_list), axis = 0)
        ol_lnxto_total_pos_ind = np.sum(np.array(final_np_ol_lnxto_array_list), axis = 0)
        ed_df['os_sl_total_pos_ind'] = pd.Series(os_sl_total_pos_ind, index=ed_df.index)
        ed_df['os_snxto_total_pos_ind'] = pd.Series(os_snxto_total_pos_ind, index=ed_df.index)
        ed_df['ol_lnxto_total_pos_ind'] = pd.Series(ol_lnxto_total_pos_ind, index=ed_df.index)
        ed_df['FinalSettleLastTradeSelect'] = ed_df['SettleLastTradeSelect'].mul(ed_df['os_sl_total_pos_ind']).\
            add(ed_df['os_sl_total_pos_ind'].abs().mul(execution_slippage))
        ed_df['FinalSettleNextOpenTradeSelect'] = ed_df['SettleNextOpenTradeSelect'].\
            mul(ed_df['os_snxto_total_pos_ind']).add(ed_df['os_snxto_total_pos_ind'].abs().mul(execution_slippage))
        ed_df['FinalLastNextOpenTradeSelect'] = ed_df['LastNextOpenTradeSelect'].\
            mul(ed_df['ol_lnxto_total_pos_ind']).add(ed_df['ol_lnxto_total_pos_ind'].abs().mul(execution_slippage))
        ed_df.os_sl_total_pos_ind.plot(title='Open-Settle/Settle-Last Total Pos Ind')
        plt.show()
        ed_df.os_snxto_total_pos_ind.plot(title='Open-Settle/Settle-NextOpen Total Pos Ind')
        plt.show()
        ed_df.ol_lnxto_total_pos_ind.plot(title='Open-Last/Last-NextOpen Total Pos Ind')
        plt.show()
        ed_df.FinalSettleLastTradeSelect.cumsum().plot(title='Settle-to-Last Cumm Pnl')
        plt.show()
        ed_df.FinalSettleNextOpenTradeSelect.cumsum().plot(title='Settle-to-NextOpen Cumm Pnl')
        plt.show()
        ed_df.FinalLastNextOpenTradeSelect.cumsum().plot(title='Last-to-NextOpen Cumm Pnl')
        plt.show()
        ed_df.to_csv('/Users/traderghazy/workspace/data/ed_df.csv')
        data = ed_df[['os_sl_lagged_corr_series', 'os_snxto_lagged_corr_series', 'ol_lnxto_lagged_corr_series',
                      'SettleLastTradeSelect', 'SettleNextOpenTradeSelect', 'LastNextOpenTradeSelect',
                      'rolling_reversion_trade_pnl', 'fwd_looking_rolling_reversion_trade_pnl',
                      'rolling_reversion_settleNextOpen_trade_pnl',
                      'fwd_looking_rolling_reversion_settleNextOpen_trade_pnl',
                      'rolling_reversion_lastNextOpen_trade_pnl',
                      'fwd_looking_rolling_reversion_lastNextOpen_trade_pnl']].dropna()
        """ the correl_filter is the conditions for filtering the correlations
            Make sure the last item in this list is either operation.and_ or operator.or_...
            this will tell the filter how to combine the conditions.
        """
        p_scat_1, p_scat_2, p_scat_3, p_correl_line = ExtendBokeh.bokeh_ed_ir_rolling_ticks_correl(data,
                                                             title=['OS-SL Rolling Cum. Sum vs. Correl',
                                                                    'OS-SL Rolling Fwd Cum. Sum vs. Correl',
                                                                    'OS-SL Point Value vs. Correl',
                                                                    'OS-SL Correlation vs. Datetime'],
                                                             subtitle=['', '', '', ''],
                                                             diff_types_to_correlate='os_sl',
                                                             type_list=['rolling_reversion_trade_pnl',
                                                                        'fwd_looking_rolling_reversion_trade_pnl',
                                                                        'SettleLastTradeSelect',
                                                                        'os_sl_lagged_corr_series'],
                                                             rolling_window_size=rolling_window_size,
                                                             correl_filter=correl_filter)
        p_scat_4, p_scat_5, p_scat_6, p_os_snxto_correl_line = ExtendBokeh.\
            bokeh_ed_ir_rolling_ticks_correl(data, title=['OS-SNXTO Rolling Cum. Sum vs. Correl',
                                                          'OS-SNXTO Rolling Fwd Cum. Sum vs. Correl',
                                                          'OS-SNXTO Point Value vs. Correl',
                                                          'OS-SNXTO Correlation vs. Datetime'],
                                             subtitle=['', '', '', ''], diff_types_to_correlate='os_snxto',
                                             type_list=['rolling_reversion_settleNextOpen_trade_pnl',
                                                        'fwd_looking_rolling_reversion_settleNextOpen_trade_pnl',
                                                        'SettleNextOpenTradeSelect',
                                                        'os_snxto_lagged_corr_series'],
                                             rolling_window_size=rolling_window_size, correl_filter=correl_filter)
        p_scat_7, p_scat_8, p_scat_9, p_ol_lnxto_correl_line = ExtendBokeh.\
            bokeh_ed_ir_rolling_ticks_correl(data,title=['OL-LNXTO Rolling Cum. Sum vs. Correl',
                                                         'OL-LNXTO Rolling Fwd Cum. Sum vs. Correl',
                                                         'OL-LNXTO Point Value vs. Correl',
                                                         'OL-LNXTO Correlation vs. Datetime'],
                                             subtitle=['', '', '', ''], diff_types_to_correlate='ol_lnxto',
                                             type_list=['rolling_reversion_lastNextOpen_trade_pnl',
                                                        'fwd_looking_rolling_reversion_lastNextOpen_trade_pnl',
                                                        'LastNextOpenTradeSelect',
                                                        'ol_lnxto_lagged_corr_series'],
                                             rolling_window_size=rolling_window_size, correl_filter=correl_filter)
        the_plots = [p_scat_1, p_scat_2, p_scat_3, p_correl_line,
                     p_scat_4, p_scat_5, p_scat_6, p_os_snxto_correl_line,
                     p_scat_7, p_scat_8, p_scat_9, p_ol_lnxto_correl_line]
        html_output_file_path = OSMuxImpl.get_proper_path('/workspace/data/bokeh/html/')
        html_output_file_title = ir_class + '_' + contract + ".scatter.html"
        html_output_file = html_output_file_path + html_output_file_title
        ExtendBokeh.show_hist_plots(the_plots,
                                    html_output_file,
                                    html_output_file_title)

        # ok, so next steps:
        # 1. correlaiton (rolling) is a stationary time series.
        # 2. We can use ARIMA to anticipate the next values. We have the searborn
        #    histograms, we can see the normality present.
        # 3. We can plot the acf and show some more stuff for the purpose of
        #    presentation.
        # 4. Once we can predict the next correlation value, we can the use the
        #   face that correlation is a predictor for next returns
        # what else? How about this...
        # 1. upon receiving a signal based on correlation value, track
        #    the rolling pnl at each subsequent correlation
        # 2. create a histogram by binning the subsequent corelation values,
        #    and plotting returns.
        # 3. the idea is that we want to see when rolling cumm pnl is maxed out
        #    based on where correlation goes after the signal is triggered.

    def prepare_post_signal_analysis_data(self,
                                          df):

        return 1

    def intraday_ir_correlation(self,
                                ir_class='EURODOLLARS',
                                contract='ED4_WHITE',
                                observation_start='2014-06-01',
                                observation_end=pd.datetime.now().strftime('%Y-%m-%d'),
                                which_lag=1):

        qdo_eurodollar = QuandlDataObject(ir_class,
                                          contract,
                                          '.csv')
        ed_df = qdo_eurodollar.get_df()
        ed_df = ed_df[observation_start:observation_end]
        ed_df = ed_df[ed_df.Volume > 0.0]
        ed_df['OpenSettleDelta'] = ed_df.Settle - ed_df.Open
        ed_df['SettleLastDelta'] = ed_df.Settle - ed_df.Last
        ed_df['LastNextOpenDelta'] = ed_df.Last -  ed_df.Open.shift(periods=-which_lag)
        os_sl_corr = np.corrcoef(ed_df.OpenSettleDelta, ed_df.SettleLastDelta)[0][1]
        sl_lnxto_corr = np.corrcoef(ed_df.SettleLastDelta[0:len(ed_df.SettleLastDelta) - which_lag].values,
                                    ed_df.LastNextOpenDelta.dropna().values)[0][1]
        os_lnxto_corr = np.corrcoef(ed_df.OpenSettleDelta[0:len(ed_df.OpenSettleDelta) - which_lag],
                                    ed_df.LastNextOpenDelta.dropna().values)[0][1]

        self.logger.info("FredApi.intraday_ir_correlation(): the correlation between open-settle delta price "
                         "and settle-last delta price is %4.6f", os_sl_corr)
        self.logger.info("FredApi.intraday_ir_correlation(): the correlation between settle-last delta price "
                         "and last-nextOpen is %4.6f", sl_lnxto_corr)
        self.logger.info("FredApi.intraday_ir_correlation(): the correlation between open-settle delta price "
                         "and last-nextOpen is %4.6f", os_lnxto_corr)

        return (os_sl_corr, sl_lnxto_corr, os_lnxto_corr)

    def create_open_settle_last_ed_ts(self,
                                      row):

        open_price = row['Open']
        settle_price = row['Settle']
        last_price = row['Last']
        settle_price_datetime=row.name.replace(hour = 14)
        open_price_datetime=row.name.replace(hour=0)
        last_price_datetime=row.name.replace(hour=16)
        px_dict = {open_price_datetime:open_price,
                   settle_price_datetime:settle_price,
                   last_price_datetime:last_price}
        return pd.Series(px_dict)

    def regress_returns(self,
                        x_series_id,
                        y_series_id):

        x_px_series = self.get_data(series_id=x_series_id)
        y_px_series = self.get_data(series_id=y_series_id)
        df = pd.concat([x_px_series, y_px_series], axis=1, join='inner').dropna()
        df_daily_rets = df.pct_change().dropna()
        # Compute correlation of x and y
        x_rets = df_daily_rets.iloc[:,0]
        y_rets = df_daily_rets.iloc[:,1]
        correlation = x_rets.corr(y_rets)
        self.logger.info("The correlation between x and y is %4.2f", correlation)
        # Convert the Series x to a DataFrame and name the column x
        df_x = pd.DataFrame(x_rets)
        # Add a constant to the DataFrame x
        df_x = sm.add_constant(df_x, 1)
        # Fit the regression of y on x
        result = sm.OLS(y_rets, df_x).fit()
        # Print out the results and look at the relationship between R-squared and the correlation above
        self.logger.info("FredApi.regress_returns():Regression Results: %s", result.summary())


if __name__ == '__main__':

    fred_obj = FredApi()
    #data = fred_obj.get_data(series_id='WILLLRGCAP',
    #                  observation_start='2018-01-01')
    #print (data.head())
    #data = fred_obj.search_fred_by_category(category_id=32413)
    #return_series = fred_obj.get_all_series_in_category(category_id=32413,
    #                                                    observation_start='2010-01-01')
    #joined = fred_obj.get_multiple_categories_series(category_id_list=[32413, 32255],
    #                                                 limit_list=[2,2])
    #fred_obj.correlation_analysis(joined)
    #fred_obj.category_series_to_csv(32255,
    #                                '/Users/traderghazy/workspace/data/fred/stock_market_indexes.csv')

    #x_series = 'SP500' # these are Fred Series ID
    #y_series = 'RU2000PR' # these are Fred Series ID
    #fred_obj.regress_returns(x_series,
    #                         y_series)
    #fred_obj.interest_rates_autocorrelation(which_lag=3)
    #fred_obj.intraday_ir_correlation(which_lag=1)
    fred_obj.rolling_eurodollar_os_sl_corr(ir_class="EURODOLLARS",
                                           contract='ED4_WHITE')
    """
    rolling_eurodollar_os_sl_corr(self,
                                  ir_class="EURODOLLARS",
                                  contract='ED4_WHITE',
                                  observation_start='2014-06-01',
                                  observation_end=pd.datetime.now().strftime('%Y-%m-%d'),
                                  which_lag=1,
                                  rolling_window_size=20,
                                  rolling_pnl_window_size=60):
    """