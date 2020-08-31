import re
from root.nested import get_logger
from root.nested.dataAccess.WebApi.TreasuryGov import nominal_yield_curve
from root.nested.dataAccess.WebApi.TreasuryGov import real_yield_curve
from root.nested.futures import cont_fut
from root.nested.dataAccess.WebApi import cboe
from root.nested.interestRateModeling.interest_rates import InterestRateInstrument, TimeValueMoney

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, DECIMAL, DateTime
from sqlalchemy.sql import text

from root.nested.visualize.extend_bokeh_datatables import ExtendDataTable
from root.nested.SysOs.os_mux import OSMuxImpl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from root.nested import get_logger

sns.set()
rcParams.update({'figure.autolayout': True})
LOGGER = get_logger()
YIELD_DIFF_RETS_PERIODS = [1, 2, 5, 10, 20, 40, 60, 120, 252]
PAR_BOND_PRICE = 1000.0
TRACK_INDEX_NOMINAL_POINT_YIELD_SPREAD_CSV = "workspace/data/treasurygov/analysis/"
HTML_FILES_DIR = "workspace/data/bokeh/html/"
SEABORN_PNG_FILES_DIR = "/workspace/data/seaborn/png/"
TRACK_INDEX_NOMINAL_POINT_YIELD_SPREAD_CSV = \
    OSMuxImpl.get_proper_path(TRACK_INDEX_NOMINAL_POINT_YIELD_SPREAD_CSV) + "tpyls.csv"
""" Get bond price from YTM """


def bond_price(par, T, ytm, coup, freq=2):
    """
    BROKEN FUNCTION, DO NOT USE!
    :param par:
    :param T:
    :param ytm:
    :param coup:
    :param freq:
    :return:
    """
    freq = float(freq)
    periods = T * freq
    coupon = coup / 100. * par / freq
    dt = [(i + 1) / freq for i in range(int(periods))]
    price = sum([coupon / (1 + ytm / freq) ** (freq * t) for t in dt]) + \
            par / (1 + ytm / freq) ** (freq * T)
    return price


def yield_delta_to_price_delta(coupon,
                               ytm,
                               maturity,
                               yield_bp_delta=1,
                               par=PAR_BOND_PRICE,
                               price=PAR_BOND_PRICE,
                               payment_freq=2):
    """
    use this function to calculate the change in price of a bond based on a
    change in yield of n basis points
    :param yield_bp_delta:
    :param payment_freq:
    :param coupon:
    :param ytm:
    :param maturity:
    :param par:
    :param price:
    :return:
    """
    LOGGER.info("yield_curve_risk_pricer.yield_delta_to_price_delta(): parameters to func %s", locals())
    ytm /= 100.0
    coupon /= 100.0
    coupon_payment = (par * coupon) / float(payment_freq)
    # increase ytm by 1 basis point
    bp_inc = yield_bp_delta / 100.0 / 100.0
    ytm_plus_one = (ytm) / payment_freq
    numer = 1 - (1 / pow((1 + ytm_plus_one), maturity * payment_freq))
    denom = ytm_plus_one
    quotient = (coupon_payment * numer) / denom
    pv_par_value = par / pow((1 + ytm_plus_one), maturity * payment_freq)
    value_of_bond_after_bp_delta = quotient + pv_par_value
    px_value_of_yield_change = price - value_of_bond_after_bp_delta
    percent_change_px = px_value_of_yield_change / price
    return value_of_bond_after_bp_delta, price, percent_change_px, px_value_of_yield_change


def vectorized_price_from_ytm(row):
    non_diffs = list(filter(lambda x: 'DIFF' not in x, row.index.tolist()))
    for prod in non_diffs:
        diff_col = prod + '_DIFF'
        the_dur = re.findall(r'\d+\w+', prod)
        if the_dur[0].find('YEAR') is not -1:
            term_yrs = float(re.findall(r'\d+', prod)[0])
        elif the_dur[0].find('MONTH') is not -1:
            term_yrs = float(re.findall(r'\d+', prod)[0]) / 12.0
        obs_ytm = row.loc[prod]
        obs_ytm_diff = row.loc[diff_col]
        if pd.isna(obs_ytm_diff):
            LOGGER.info("yield_curve_risk_pricer.vectorized_price_from_ytm(): "
                        "default price being used for %s bond price", prod)
        elif int(term_yrs * 1000000.0) < int(1.0 * 1000000.0):
            # these are t-bills, they don't have a coupon.
            return_tuple = yield_delta_to_price_delta(coupon=0.0,
                                                      ytm=obs_ytm,
                                                      maturity=term_yrs,
                                                      yield_bp_delta=obs_ytm_diff,
                                                      par=PAR_BOND_PRICE,
                                                      price=global_bond_prices[prod])
            global_bond_prices[prod] = return_tuple[0]
        else:
            return_tuple = yield_delta_to_price_delta(coupon=static_coupon_dict[prod],
                                                      ytm=obs_ytm,
                                                      maturity=term_yrs,
                                                      yield_bp_delta=int(obs_ytm_diff * 100.0),
                                                      par=PAR_BOND_PRICE,
                                                      price=global_bond_prices[prod])
            global_bond_prices[prod] = return_tuple[0]
        row[prod + '_UPDATED_PRICE'] = global_bond_prices[prod]
    return row


def track_index_nominal_point_yield_spread(duration_point,
                                           query_start_date,
                                           query_end_date,
                                           risk_asset_list):
    """
    track a spread price of risk asset versus a specific
    nominal yield on the curve, not a spread.
    :param query_start_date:
    :param query_end_date:
    :param risk_asset:
    :param duration_point:
    :return:
    """
    from itertools import combinations
    global global_bond_prices
    global static_coupon_dict
    global_bond_prices = dict()
    static_coupon_dict = dict()
    yc_ts = nominal_yield_curve.get_yield_point_ts_from_db(query_start_date=query_start_date,
                                                           query_end_date=query_end_date,
                                                           points_on_curve=duration_point)
    for col_nm in yc_ts.columns:
        global_bond_prices[col_nm] = PAR_BOND_PRICE
        yc_ts[col_nm + '_DIFF'] = yc_ts.diff(periods=1)
        static_coupon_dict[col_nm] = yc_ts.iloc[0][col_nm]
    price_yc_ts = yc_ts.apply(vectorized_price_from_ytm, axis=1)
    updated_px_cols = list(filter(lambda x: '_UPDATED_PRICE' in x, price_yc_ts.columns))
    for up_px_col in updated_px_cols:
        for period_len in YIELD_DIFF_RETS_PERIODS:
            price_yc_ts[up_px_col + '_' + str(period_len) + 'D_PCT_CHANGE'] = \
                price_yc_ts[up_px_col].pct_change(periods=period_len)
            price_yc_ts[up_px_col + '_' + str(period_len) + 'D_DIFF'] = price_yc_ts[up_px_col].diff(periods=period_len)
    price_yc_ts.to_csv(path_or_buf=TRACK_INDEX_NOMINAL_POINT_YIELD_SPREAD_CSV + '.bonds.csv')
    comb = combinations(range(len(duration_point)), r=2)
    yc_ts = price_yc_ts
    for comb_tuple in comb:
        yc_ts[yc_ts.columns[comb_tuple[1]] + '-' + yc_ts.columns[comb_tuple[0]]] = \
            yc_ts[yc_ts.columns[comb_tuple[1]]] - yc_ts[yc_ts.columns[comb_tuple[0]]]
    for poc in yc_ts.columns:
        if poc.find('-') is -1 and poc.find('DIFF') is -1 and poc.find('PRICE') is -1 and poc.find('PCT_CHANGE') is -1:
            # this is a single point series.
            for period_len in YIELD_DIFF_RETS_PERIODS:
                yc_ts[poc + '_' + str(period_len) + 'D_DIFF'] = yc_ts[poc].diff(periods=period_len)
                yc_ts[poc + '_' + str(period_len) + 'D_PCT_CHANGE'] = yc_ts[poc].pct_change(periods=period_len)
    risk_asset_ts = cboe.get_cboe_sym_ts_from_db(query_start_date=query_start_date,
                                                 query_end_date=query_end_date,
                                                 symbol_list=risk_asset_list)
    risk_asset_ts = risk_asset_ts.unstack(level=-1)
    risk_asset_ts_close_px = risk_asset_ts['Close']
    for risk_asset in risk_asset_list:
        for period_len in YIELD_DIFF_RETS_PERIODS:
            risk_asset_ts_close_px[risk_asset + '_' + str(period_len) + 'D_DIFF'] = \
                risk_asset_ts_close_px[risk_asset].diff(periods=period_len)
            risk_asset_ts_close_px[risk_asset + '_' + str(period_len) + 'D_PCT_CHANGE'] = \
                risk_asset_ts_close_px[risk_asset].pct_change(periods=period_len)
    risk_close_px_merge_curve = pd.merge(left=yc_ts, right=risk_asset_ts_close_px, left_index=True, right_index=True)
    risk_asset_ts_open_px = risk_asset_ts['Open']
    risk_asset_ts_high_px = risk_asset_ts['High']
    risk_asset_ts_low_px = risk_asset_ts['Low']
    risk_close_px_merge_curve.to_csv(path_or_buf=TRACK_INDEX_NOMINAL_POINT_YIELD_SPREAD_CSV)
    spread_df = correlation_heatmaps(risk_close_px_merge_curve, duration_point, risk_asset_list)
    return spread_df


def correlation_heatmaps(df, duration_point, risk_asset_list):
    LOGGER.info("yield_curve_risk_pricer.make_spread(): running make_spread() function...")
    col_list = df.columns
    risk_off_diffs_list = []
    risk_off_pct_change_s_list = []
    risk_off_derived_px_diffs_list = []
    risk_off_derived_px_pctChange_list = []
    for dp in duration_point:
        label = nominal_yield_curve.DURATION_TO_DBCOL_MAPPING[dp]
        for ydr_per in YIELD_DIFF_RETS_PERIODS:
            risk_off_diffs = label + '_' + str(ydr_per) + 'D_DIFF'
            risk_off_pct_change_s = label + '_' + str(ydr_per) + 'D_PCT_CHANGE'
            risk_off_derived_px_diffs = label + '_UPDATED_PRICE_' + str(ydr_per) + 'D_DIFF'
            risk_off_derived_px_pctChange = label + '_UPDATED_PRICE_' + str(ydr_per) + 'D_PCT_CHANGE'
            risk_off_diffs_list.append(risk_off_diffs)
            risk_off_pct_change_s_list.append(risk_off_pct_change_s)
            risk_off_derived_px_diffs_list.append(risk_off_derived_px_diffs)
            risk_off_derived_px_pctChange_list.append(risk_off_derived_px_pctChange)
        diffs_intersection_of_lists = \
            list(sorted(set(risk_off_diffs_list).intersection(col_list)))
        pct_change_s_intersection_of_lists = \
            list(sorted(set(risk_off_pct_change_s_list).intersection(col_list)))
        derived_px_diffs_intersection_of_lists = \
            list(sorted(set(risk_off_derived_px_diffs_list).intersection(col_list)))
        derived_px_pctChange_intersection_of_lists = \
            list(sorted(set(risk_off_derived_px_pctChange_list).intersection(col_list)))
        for risk_on in risk_asset_list:
            risk_on_diffs_list = []
            risk_on_pct_change_s_list = []
            for ydr_per in YIELD_DIFF_RETS_PERIODS:
                risk_on_diffs = risk_on + '_' + str(ydr_per) + 'D_DIFF'
                risk_on_pct_change_s = risk_on + '_' + str(ydr_per) + 'D_PCT_CHANGE'
                risk_on_diffs_list.append(risk_on_diffs)
                risk_on_pct_change_s_list.append(risk_on_pct_change_s)
            diffs_intersection_risk_on_list = list(sorted(set(risk_on_diffs_list).intersection(col_list)))
            pct_change_s_intersection_risk_on_list = list(sorted(set(risk_on_pct_change_s_list).intersection(col_list)))
            corr_matrix_diffRiskOff_diffRiskOn = \
                df[diffs_intersection_of_lists + diffs_intersection_risk_on_list].corr()
            corr_matrix_diffRiskOff_pctChangeRiskOn = \
                df[diffs_intersection_of_lists + pct_change_s_intersection_risk_on_list].corr()
            corr_matrix_pctChangeRiskOff_diffRiskOn = \
                df[pct_change_s_intersection_of_lists + diffs_intersection_risk_on_list].corr()
            corr_matrix_pctChangeRiskOff_pctChangeRiskOn = \
                df[pct_change_s_intersection_of_lists + pct_change_s_intersection_risk_on_list].corr()

            # derived px diff/pct_change versus risk-on diff/pctChange
            corr_matrix_derivedPxDiffRiskOff_diffRiskOn = \
                df[derived_px_diffs_intersection_of_lists + diffs_intersection_risk_on_list].corr()
            corr_matrix_derivedPxDiffRiskOff_pctChangeRiskOn = \
                df[derived_px_diffs_intersection_of_lists + pct_change_s_intersection_risk_on_list].corr()
            corr_matrix_derivedPxPctChangeRiskOff_diffRiskOn = \
                df[derived_px_pctChange_intersection_of_lists + diffs_intersection_risk_on_list].corr()
            corr_matrix_derivedPxPctChangeRiskOff_pctChangeRiskOn = \
                df[derived_px_pctChange_intersection_of_lists + pct_change_s_intersection_risk_on_list].corr()

            plt.figure(figsize=(27, 18))
            LOGGER.info("yield_curve_risk_pricer.correlation_heatmap(): "
                        "corr_matrix_derivedPxPctChangeRiskOff_pctChangeRiskOn")
            ax = sns.heatmap(corr_matrix_derivedPxPctChangeRiskOff_pctChangeRiskOn, annot=True, linewidths=0.5)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=12)
            ax.figure.savefig(OSMuxImpl.get_proper_path(SEABORN_PNG_FILES_DIR) +
                              "corr_matrix_derivedPxPctChangeRiskOff_pctChangeRiskOn.png")
            # plt.tight_layout()
            top_correl_df = find_top_correlations(corr_matrix_derivedPxPctChangeRiskOff_pctChangeRiskOn)
            print (top_correl_df)
            make_spread_price(top_correl_df, df)
            return

            plt.figure(figsize=(27, 18))
            LOGGER.info("yield_curve_risk_pricer.correlation_heatmap(): corr_matrix_diffRiskOff_diffRiskOn")
            ax = sns.heatmap(corr_matrix_diffRiskOff_diffRiskOn, annot=True, linewidths=0.5)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=12)
            ax.figure.savefig(
                OSMuxImpl.get_proper_path(SEABORN_PNG_FILES_DIR) + "corr_matrix_diffRiskOff_diffRiskOn.png")
            top_correl_df = find_top_correlations(corr_matrix_diffRiskOff_diffRiskOn)
            make_spread_price(top_correl_df, df)

            plt.figure(figsize=(27, 18))
            LOGGER.info("yield_curve_risk_pricer.correlation_heatmap(): corr_matrix_diffRiskOff_pctChangeRiskOn")
            ax = sns.heatmap(corr_matrix_diffRiskOff_pctChangeRiskOn, annot=True, linewidths=0.5)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=12)
            # plt.tight_layout()
            ax.figure.savefig(
                OSMuxImpl.get_proper_path(SEABORN_PNG_FILES_DIR) + "corr_matrix_diffRiskOff_pctChangeRiskOn.png")
            find_top_correlations(corr_matrix_diffRiskOff_pctChangeRiskOn)

            plt.figure(figsize=(27, 18))
            LOGGER.info("yield_curve_risk_pricer.correlation_heatmap(): corr_matrix_pctChangeRiskOff_diffRiskOn")
            ax = sns.heatmap(corr_matrix_pctChangeRiskOff_diffRiskOn, annot=True, linewidths=0.5)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=12)
            # plt.tight_layout()
            ax.figure.savefig(
                OSMuxImpl.get_proper_path(SEABORN_PNG_FILES_DIR) + "corr_matrix_pctChangeRiskOff_diffRiskOn.png")
            find_top_correlations(corr_matrix_pctChangeRiskOff_diffRiskOn)

            plt.figure(figsize=(27, 18))
            LOGGER.info("yield_curve_risk_pricer.correlation_heatmap(): corr_matrix_pctChangeRiskOff_pctChangeRiskOn")
            ax = sns.heatmap(corr_matrix_pctChangeRiskOff_pctChangeRiskOn, annot=True, linewidths=0.5)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=12)
            # plt.tight_layout()
            ax.figure.savefig(
                OSMuxImpl.get_proper_path(SEABORN_PNG_FILES_DIR) + "corr_matrix_pctChangeRiskOff_pctChangeRiskOn.png")
            find_top_correlations(corr_matrix_pctChangeRiskOff_pctChangeRiskOn)

            plt.figure(figsize=(27, 18))
            LOGGER.info("yield_curve_risk_pricer.correlation_heatmap(): corr_matrix_derivedPxDiffRiskOff_diffRiskOn")
            ax = sns.heatmap(corr_matrix_derivedPxDiffRiskOff_diffRiskOn, annot=True, linewidths=0.5)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=12)
            ax.figure.savefig(
                OSMuxImpl.get_proper_path(SEABORN_PNG_FILES_DIR) + "corr_matrix_derivedPxDiffRiskOff_diffRiskOn.png")
            # plt.tight_layout()
            find_top_correlations(corr_matrix_derivedPxDiffRiskOff_diffRiskOn)

            plt.figure(figsize=(27, 18))
            LOGGER.info("yield_curve_risk_pricer.correlation_heatmap(): "
                        "corr_matrix_derivedPxDiffRiskOff_pctChangeRiskOn")
            ax = sns.heatmap(corr_matrix_derivedPxDiffRiskOff_pctChangeRiskOn, annot=True, linewidths=0.5)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=12)
            ax.figure.savefig(
                OSMuxImpl.get_proper_path(SEABORN_PNG_FILES_DIR) + "corr_matrix_derivedPxDiffRiskOff_pctChangeRiskOn"
                                                                   ".png")
            # plt.tight_layout()
            find_top_correlations(corr_matrix_derivedPxDiffRiskOff_pctChangeRiskOn)

            plt.figure(figsize=(27, 18))
            LOGGER.info("yield_curve_risk_pricer.correlation_heatmap(): "
                        "corr_matrix_derivedPxPctChangeRiskOff_diffRiskOn")
            ax = sns.heatmap(corr_matrix_derivedPxPctChangeRiskOff_diffRiskOn, annot=True, linewidths=0.5)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=12)
            ax.figure.savefig(
                OSMuxImpl.get_proper_path(SEABORN_PNG_FILES_DIR) + "corr_matrix_derivedPxChangeRiskOff_diffRiskOn.png")
            # plt.tight_layout()
            find_top_correlations(corr_matrix_derivedPxPctChangeRiskOff_diffRiskOn)


def vectorized_make_spread_price(row, price_df):
    from root.nested.futures.cont_fut import GraphicalTimeSeries
    spread_price = price_df[row['Target']].subtract(price_df[row['Hedge']])
    spread_price_df = spread_price.to_frame().dropna()
    spread_price_df.columns = [row['Target'] + '-' + row['Hedge']]
    spread_price_df.index.rename('Date', inplace=True)
    #gts = GraphicalTimeSeries(plot_data={spread_price_df.columns[0]: spread_price_df})
    #gts.multiple_line_plot_bokeh(combined_plot=False)
    spread_price_df.plot(kind='line', y=spread_price_df.columns[0], color='red')

    #print(spread_price_df.head(10))
    #sns.tsplot(data=spread_price_df, time='Date', value=(row['Target'] + '-' + row['Hedge']))
    plt.show()


def make_spread_price(top_correl_df, original_price_df):
    result_df = top_correl_df.apply(vectorized_make_spread_price, axis=1, price_df=original_price_df)


def find_top_correlations(corr_matrix, upper_th_correl=0.6, lower_th_correl=-0.6):
    LOGGER.info("yield_curve_risk_pricer.find_top_correlations(): "
                "running find_top_correlations(upper_th_correl=%s, lower_th_correl=%s)...",
                str(upper_th_correl), str(lower_th_correl))
    # the matrix is symmetric so we need to extract upper triangle matrix without diagonal (k = 1)
    sol = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).
                             astype(np.bool)).stack().sort_values(ascending=False))
    sol = sol[(sol > upper_th_correl) | (sol < lower_th_correl)]
    multi_index_df = sol.index.set_names(['Target', 'Hedge'])
    df = pd.DataFrame(data=sol.values, index=multi_index_df)
    df.columns = ["Correlation"]
    bokeh_document = ExtendDataTable.make_correlation_datatable(df)
    ExtendDataTable.validate_show_document(html_document=bokeh_document,
                                           html_filename="topCorrels.html",
                                           html_dir=HTML_FILES_DIR,
                                           viewHtml=False)
    return df


def validate_show_document(html_document, html_filename, html_dir, viewHtml=False):
    from bokeh.embed import file_html
    from bokeh.resources import INLINE
    from bokeh.util.browser import view
    print("the html document", file_html(html_document, INLINE, "CorrelTables"))
    html_document.validate()
    proper_dir = OSMuxImpl.get_proper_path(html_dir)
    proper_filename = proper_dir + html_filename
    with open(proper_filename, "w") as f:
        f.write(file_html(html_document, INLINE, "Data Tables"))
    LOGGER.info("extend_bokeh_datatables.ExtendBokeh.validate_show_document(): wrote %s in dir %s ",
                html_filename, proper_dir)
    if viewHtml is not False:
        view(proper_filename)


def calc_rolling_correlation():
    """
    in this function, do SPEARMAN RANK.
    :return:
    """
    return 1


def track_index_real_point_yield_spread(duration_point,
                                        risk_asset="SPX"):
    """
    track a spread price of a risk asset versus a specific
    real yield on the real yield curve.
    :param duration_point:
    :return:
    """

    return 1


def track_index_nominal_curve_spread(short_duration_point=3,
                                     long_duration_point=360,
                                     risk_asset="SPX"):
    """
    this will track a spread price for risk asset versus a yield spread, nominal yields.
    :param short_duration_point:
    :param long_duration_point:
    :return:
    """
    return 1


def track_index_real_curve_spread(short_duration_point=3,
                                  long_duration_point=360,
                                  risk_asset="SPX"):
    """
    this will track a spread price for risk asset versus a yield spread, real yields.
    :param short_duration_point:
    :param long_duration_point:
    :return:
    """
    return 1


def get_risk_asset_data(asset_type,
                        asset_symbol):
    return 1


def get_yield_point_ts_from_db(duration_point):
    return 1


track_index_nominal_point_yield_spread(duration_point=[24],
                                       query_start_date="2019-06-01",
                                       query_end_date="2020-06-11",
                                       risk_asset_list=['SPX', 'VIX', 'VVIX'])

# yield_to_px_delta = yield_delta_to_price_delta(coupon=0.17, ytm=0.19, maturity=2,
#                                               yield_bp_delta=2, price=1000)
# print(yield_to_px_delta)
# yield_to_px_delta = yield_delta_to_price_delta(coupon=0.17, ytm=0.19, maturity=2,
#                                               yield_bp_delta=0, price=yield_to_px_delta[0])
# print(yield_to_px_delta)
# yield_to_px_delta = yield_delta_to_price_delta(coupon=0.14, ytm=0.19, maturity=2,
#                                               yield_bp_delta=0, price=yield_to_px_delta[0])
# print(yield_to_px_delta)
# yield_to_px_delta = yield_delta_to_price_delta(coupon=0.14, ytm=0.19, maturity=2,
#                                               yield_bp_delta=0, price=yield_to_px_delta[0])
# print(yield_to_px_delta)
# yield_to_px_delta = yield_delta_to_price_delta(coupon=0.14, ytm=0.19, maturity=2,
#                                               yield_bp_delta=0, price=yield_to_px_delta[0])
# print(yield_to_px_delta)
