from root.nested import get_logger
from root.nested.futures import cont_fut
from root.nested.dataAccess.WebApi import cboe
import sys
import pandas as pd
from datetime import datetime, time, timedelta
from root.nested.dataAccess.quandl_interface import QuandlSymbolInterface
from root.nested.dataAccess.WebApi.TreasuryGov import nominal_yield_curve, real_yield_curve

LOGGER = get_logger()

# start date of data (this should be consistent/constant, regardless of run-type
start_date = str(sys.argv[1])
# end date of data (this should be consistent/constant, regardless of run_type
end_date = str(sys.argv[2])
# flag for MySQL db insert or not. Use if only needing csv files.
db_insert = eval(str.capitalize(str.lower(sys.argv[3])))  # if 0, do not do a db insert. If 1, do the insert.
all_data_years_list = [val.to_pydatetime().year for val in
                       pd.date_range(start=start_date, end=end_date, freq='A')]


def run_eurodollars(symbol="ED", run_type='DAILY'):
    """
    Run end of day eurodollars first.
    :return:
    """
    all_expirations_ed = pd.DataFrame()
    num_expirations = len(all_data_years_list) * len(cont_fut.fut_expiry_code_mapping[symbol].keys())
    if str.upper(run_type) == "DAILY":
        for expiration in range(1, 25):
            LOGGER.info("cont_fut.main(): rolling symbol %s exp %s to %s", symbol, str(expiration), str(expiration + 1))
            ed_final, returned_start_date, returned_end_date = \
                cont_fut.run_eurodollar_roll_method_daily(symbol,
                                                          "CHRIS/CME_ED" + str(expiration),
                                                          "CHRIS/CME_ED" + str(expiration + 1),
                                                          start_date,
                                                          end_date,
                                                          all_data_years_list,
                                                          num_expirations)
            if expiration == 1:
                dates = pd.date_range(start_date, ed_final.index[-1], freq='B')
                all_expirations_ed = pd.DataFrame(index=dates)
            all_expirations_ed = pd.merge(left=all_expirations_ed, right=ed_final[['MERGED_' + symbol + '_SETTLE',
                                                                                   'MERGED_' + symbol + '_OPEN',
                                                                                   'MERGED_' + symbol + '_HIGH',
                                                                                   'MERGED_' + symbol + '_LOW',
                                                                                   'MERGED_' + symbol + '_LAST']],
                                          left_index=True, right_index=True)
            all_expirations_ed.rename(
                columns={'MERGED_' + symbol + "_SETTLE": 'MERGED_' + symbol + str(expiration) + "_SETTLE",
                         'MERGED_' + symbol + "_OPEN": 'MERGED_' + symbol + str(expiration) + "_OPEN",
                         'MERGED_' + symbol + "_HIGH": 'MERGED_' + symbol + str(expiration) + "_HIGH",
                         'MERGED_' + symbol + "_LOW": 'MERGED_' + symbol + str(expiration) + "_LOW",
                         'MERGED_' + symbol + "_LAST": 'MERGED_' + symbol + str(expiration) + "_LAST"},
                inplace=True)
            cont_fut.write_to_csv(ed_final,
                                  "ED" + str(expiration),
                                  str(start_date),
                                  str(end_date),
                                  force_new_file=False)
        cont_fut.write_to_csv(all_expirations_ed, "ED_TOTAL_CURVE", str(start_date), str(end_date),
                              force_new_file=False)
        if db_insert:
            cont_fut.ed_curve_to_db(all_expirations_ed)
    elif str.upper(run_type) == "HISTORICAL":
        for expiration in range(1, 25):
            ed_final = cont_fut.run_eurodollar_roll_method(symbol,
                                                           "CHRIS/CME_ED" + str(expiration),
                                                           "CHRIS/CME_ED" + str(expiration + 1),
                                                           start_date,
                                                           end_date,
                                                           all_data_years_list,
                                                           num_expirations)
            if expiration == 1:
                dates = pd.date_range(start_date, ed_final.index[-1], freq='B')
                all_expirations_ed = pd.DataFrame(index=dates)
            all_expirations_ed = pd.merge(left=all_expirations_ed, right=ed_final[['MERGED_' + symbol + '_SETTLE',
                                                                                   'MERGED_' + symbol + '_OPEN',
                                                                                   'MERGED_' + symbol + '_HIGH',
                                                                                   'MERGED_' + symbol + '_LOW',
                                                                                   'MERGED_' + symbol + '_LAST']],
                                          left_index=True, right_index=True)
            all_expirations_ed.rename(
                columns={'MERGED_' + symbol + "_SETTLE": 'MERGED_' + symbol + str(expiration) + "_SETTLE",
                         'MERGED_' + symbol + "_OPEN": 'MERGED_' + symbol + str(expiration) + "_OPEN",
                         'MERGED_' + symbol + "_HIGH": 'MERGED_' + symbol + str(expiration) + "_HIGH",
                         'MERGED_' + symbol + "_LOW": 'MERGED_' + symbol + str(expiration) + "_LOW",
                         'MERGED_' + symbol + "_LAST": 'MERGED_' + symbol + str(expiration) + "_LAST"},
                inplace=True)
            cont_fut.write_to_csv(ed_final, "ED" + str(expiration), start_date, end_date, force_new_file=True)
        cont_fut.write_to_csv(all_expirations_ed, "ED_TOTAL_CURVE", start_date, end_date, force_new_file=True)
        if db_insert:
            cont_fut.ed_curve_to_db(all_expirations_ed)


def run_vix_futs_curve(length_of_curve=7,
                       symbol="VX",
                       provider_symbol="CHRIS/CBOE_VX",
                       run_type="DAILY"):
    #run_crude_oil(length_of_curve=length_of_curve,
    #              symbol=symbol,
    #              provider_symbol=provider_symbol,
    #              run_type=run_type)
    unadjusted_csv_file = cont_fut.vx_continuous_contract_creation()
    list_cont_contract_dfs = cont_fut.vx_roll_adjust_from_csv(unadj_csv_file=unadjusted_csv_file)
    single_df_cont_contracts, csv_file_cont_contracts = cont_fut.merge_vx_futs_dataframes(list_cont_contract_dfs)
    if db_insert:
        cont_fut.vx_curve_to_db(df=single_df_cont_contracts)


def run_crude_oil(length_of_curve=38,
                  symbol="CL",
                  provider_symbol="CHRIS/CME_CL",
                  run_type="DAILY"):
    """
    The default here is for crude oil curve, but we can use this function to also run the VX curve because
    it is a MONTHLY expiry contract, just like CL.
    :param provider_symbol:
    :param length_of_curve:
    :param symbol:
    :param run_type:
    :return:
    """
    # we want to do expiration 1 thru 39 to create the curve. Quandl data goes out to 39th expiration.
    # not much open interest beyond 36.
    # CL Futures (only use proportional roll?)
    all_expirations_cl = pd.DataFrame()
    num_expirations = len(all_data_years_list) * len(cont_fut.fut_expiry_code_mapping[symbol].keys())
    if str.upper(run_type) == "DAILY":
        for expiration in range(1, length_of_curve):
            LOGGER.info("cont_fut.main(): rolling symbol %s exp %s to %s", symbol, str(expiration), str(expiration + 1))
            cl_final, returned_start_date, returned_end_date = \
                cont_fut.run_proportional_roll_method_daily(symbol,
                                                            provider_symbol + str(expiration),
                                                            provider_symbol + str(expiration + 1),
                                                            start_date,
                                                            end_date,
                                                            all_data_years_list,
                                                            num_expirations)
            if expiration == 1:
                dates = pd.date_range(start_date, cl_final.index[-1], freq='B')
                all_expirations_cl = pd.DataFrame(index=dates)
            all_expirations_cl = pd.merge(left=all_expirations_cl, right=cl_final[['MERGED_' + symbol + '_SETTLE',
                                                                                   'MERGED_' + symbol + '_OPEN',
                                                                                   'MERGED_' + symbol + '_HIGH',
                                                                                   'MERGED_' + symbol + '_LOW',
                                                                                   'MERGED_' + symbol + '_LAST']],
                                          left_index=True, right_index=True)
            all_expirations_cl.rename(
                columns={'MERGED_' + symbol + "_SETTLE": 'MERGED_' + symbol + str(expiration) + "_SETTLE",
                         'MERGED_' + symbol + "_OPEN": 'MERGED_' + symbol + str(expiration) + "_OPEN",
                         'MERGED_' + symbol + "_HIGH": 'MERGED_' + symbol + str(expiration) + "_HIGH",
                         'MERGED_' + symbol + "_LOW": 'MERGED_' + symbol + str(expiration) + "_LOW",
                         'MERGED_' + symbol + "_LAST": 'MERGED_' + symbol + str(expiration) + "_LAST"},
                inplace=True)
            cont_fut.write_to_csv(cl_final,
                                  symbol + str(expiration),
                                  str(start_date),
                                  str(end_date),
                                  force_new_file=False)
        cont_fut.write_to_csv(all_expirations_cl, symbol + "_TOTAL_CURVE", str(start_date), str(end_date),
                              force_new_file=False)
        if db_insert and symbol == 'CL':
            cont_fut.cl_curve_to_db(all_expirations_cl)
        elif db_insert and symbol == 'VX':
            cont_fut.vx_curve_to_db(all_expirations_cl)
    elif str.upper(run_type) == "HISTORICAL":
        for expiration in range(1, length_of_curve):
            cl_final = cont_fut.run_proportional_roll_method(symbol,
                                                             provider_symbol + str(expiration),
                                                             provider_symbol + str(expiration + 1),
                                                             start_date,
                                                             end_date,
                                                             all_data_years_list,
                                                             num_expirations)
            if expiration == 1:
                dates = pd.date_range(start_date, cl_final.index[-1], freq='B')
                all_expirations_cl = pd.DataFrame(index=dates)
            all_expirations_cl = pd.merge(left=all_expirations_cl, right=cl_final[['MERGED_' + symbol + '_SETTLE',
                                                                                   'MERGED_' + symbol + '_OPEN',
                                                                                   'MERGED_' + symbol + '_HIGH',
                                                                                   'MERGED_' + symbol + '_LOW',
                                                                                   'MERGED_' + symbol + '_LAST']],
                                          left_index=True, right_index=True)
            # Should we drop entire row when one of the continuous contracts has a missing data?
            # or should we just leave it as NA?
            # or should we forward fill it or use some method?
            # right now we will default to just dropping the entire row. We will change if there are a lot of NAs.
            all_expirations_cl.dropna(inplace=True)
            all_expirations_cl.rename(
                columns={'MERGED_' + symbol + "_SETTLE": 'MERGED_' + symbol + str(expiration) + "_SETTLE",
                         'MERGED_' + symbol + "_OPEN": 'MERGED_' + symbol + str(expiration) + "_OPEN",
                         'MERGED_' + symbol + "_HIGH": 'MERGED_' + symbol + str(expiration) + "_HIGH",
                         'MERGED_' + symbol + "_LOW": 'MERGED_' + symbol + str(expiration) + "_LOW",
                         'MERGED_' + symbol + "_LAST": 'MERGED_' + symbol + str(expiration) + "_LAST"},
                inplace=True)
            cont_fut.write_to_csv(cl_final, symbol + str(expiration), start_date, end_date, force_new_file=True)
        cont_fut.write_to_csv(all_expirations_cl, symbol + "_TOTAL_CURVE", start_date, end_date, force_new_file=True)
        if db_insert and symbol == 'CL':
            cont_fut.cl_curve_to_db(all_expirations_cl)
        elif db_insert and symbol == 'VX':
            cont_fut.vx_curve_to_db(all_expirations_cl)


def run_currencies(run_type="DAILY"):
    """
    This function will run both cash currencies and futures.
    When we run the spot mkt currencies, we w
    :param run_type:
    :return:
    """
    curr_fut_sym_dict = {'JY': 'CHRIS/CME_JY',
                         'EC': 'CHRIS/CME_EC',
                         'AD': 'CHRIS/CME_AD',
                         'CD': 'CHRIS/CME_CD',
                         'BP': 'CHRIS/CME_BP',
                         'SF': 'CHRIS/CME_SF'}
    local_fut_to_spot_mapping = {'JY': "JPY_USD_spot",
                                 'EC': "EUR_USD_spot",
                                 'AD': "AUD_USD_spot",
                                 'CD': "CAD_USD_spot",
                                 'BP': "GBP_USD_spot",
                                 'SF': "CHF_USD_spot"}
    qsi = QuandlSymbolInterface()
    if run_type == "HISTORICAL":
        for curr_fut_sym in sorted(set(curr_fut_sym_dict.keys())):
            spot_con_desc = local_fut_to_spot_mapping[curr_fut_sym]
            spot_quandl_sym = qsi.get_quandl_symbol(class_of_data="FOREX_TO_USD",
                                                    local_symbol=spot_con_desc)
            LOGGER.info("run_eod.run_currencies(): running spot quandl symbol %s", spot_quandl_sym)
            num_expirations = len(all_data_years_list) * len(cont_fut.fut_expiry_code_mapping[curr_fut_sym].keys())
            fut_quandl_sym = curr_fut_sym_dict[curr_fut_sym]
            prop_roll_futs = cont_fut.run_currency_futures_roll(curr_fut_sym,
                                                                fut_quandl_sym + "1",
                                                                fut_quandl_sym + "2",
                                                                start_date,
                                                                end_date,
                                                                all_data_years_list,
                                                                num_expirations,
                                                                roll_type='proportional')
            flat_px_roll_futs = cont_fut.run_currency_futures_roll(curr_fut_sym,
                                                                   fut_quandl_sym + "1",
                                                                   fut_quandl_sym + "2",
                                                                   start_date,
                                                                   end_date,
                                                                   all_data_years_list,
                                                                   num_expirations,
                                                                   roll_type='flatprice')
            spot_curr_df = cont_fut.run_spot_currency_roll(spot_quandl_sym,
                                                           spot_con_desc,
                                                           start_date,
                                                           end_date)
            prop_fut_and_spot = pd.merge(left=prop_roll_futs, right=spot_curr_df,
                                         left_index=True, right_index=True)
            flat_px_fut_and_spot = pd.merge(left=flat_px_roll_futs, right=spot_curr_df,
                                            left_index=True, right_index=True)
            cont_fut.write_to_csv(prop_fut_and_spot, curr_fut_sym + "_Proportional_Roll", start_date, end_date, force_new_file=True)
            cont_fut.write_to_csv(flat_px_fut_and_spot, curr_fut_sym + "_FlatPrice_Roll", start_date, end_date, force_new_file=True)
            cont_fut.currency_futures_px_to_db(prop_fut_and_spot, roll_type='proportional', isDaily=False)
            cont_fut.currency_futures_px_to_db(flat_px_fut_and_spot, roll_type='flatprice', isDaily=False)
    elif run_type == 'DAILY':
        for curr_fut_sym in sorted(set(curr_fut_sym_dict.keys())):
            spot_con_desc = local_fut_to_spot_mapping[curr_fut_sym]
            spot_quandl_sym = qsi.get_quandl_symbol(class_of_data="FOREX_TO_USD",
                                                    local_symbol=spot_con_desc)
            LOGGER.info("run_eod.run_currencies(): running spot quandl symbol %s", spot_quandl_sym)
            num_expirations = len(all_data_years_list) * len(cont_fut.fut_expiry_code_mapping[curr_fut_sym].keys())
            fut_quandl_sym = curr_fut_sym_dict[curr_fut_sym]
            prop_roll_futs, spot_start_date, spot_end_date = \
                cont_fut.run_currency_futures_roll_daily(curr_fut_sym,
                                                         fut_quandl_sym + "1",
                                                         fut_quandl_sym + "2",
                                                         start_date,
                                                         end_date,
                                                         all_data_years_list,
                                                         num_expirations,
                                                         roll_type='proportional')
            flat_px_roll_futs, spot_start_date, spot_end_date = \
                cont_fut.run_currency_futures_roll_daily(curr_fut_sym,
                                                         fut_quandl_sym + "1",
                                                         fut_quandl_sym + "2",
                                                         start_date,
                                                         end_date,
                                                         all_data_years_list,
                                                         num_expirations,
                                                         roll_type='flatprice')
            spot_curr_df = cont_fut.run_spot_currency_roll(spot_quandl_sym,
                                                           spot_con_desc,
                                                           spot_start_date,
                                                           spot_end_date)
            prop_fut_and_spot = pd.merge(left=prop_roll_futs, right=spot_curr_df, left_index=True, right_index=True)
            cont_fut.write_to_csv(prop_fut_and_spot, curr_fut_sym + "_Proportional_Roll", start_date, end_date)
            cont_fut.currency_futures_px_to_db(prop_fut_and_spot, roll_type='proportional', isDaily=True)
            flat_px_fut_and_spot = pd.merge(left=flat_px_roll_futs, right=spot_curr_df, left_index=True,
                                            right_index=True)
            cont_fut.write_to_csv(flat_px_fut_and_spot, curr_fut_sym + "_FlatPrice_Roll", start_date, end_date)
            cont_fut.currency_futures_px_to_db(flat_px_fut_and_spot, roll_type='flatprice', isDaily=True)


def run_equity_futures(run_type="DAILY"):
    # ES Futures
    if run_type == "HISTORICAL":
        es_prop_roll_futs_final = cont_fut.run_equity_futures_roll("ES", "CHRIS/CME_ES1", "CHRIS/CME_ES2", "2017-01-01",
                                                                   "2021-12-31", [2017, 2018, 2019, 2020], 16,
                                                                   roll_type='proportional')
        cont_fut.write_to_csv(es_prop_roll_futs_final, "ES_Proportional_Roll", "2017-01-01", "2021-12-31", force_new_file=True)
        # where the condition is False, meaning where there are Null values in the dataframe, change them to None
        es_prop_roll_futs_final = es_prop_roll_futs_final.where((pd.notnull(es_prop_roll_futs_final)), None)
        es_prop_roll_futs_final['BACK_MONTH_ES_OPEN'] = \
            es_prop_roll_futs_final['BACK_MONTH_ES_OPEN'].fillna(value=es_prop_roll_futs_final['BACK_MONTH_ES_OPEN'].
                                                                 shift(periods=1))
        cont_fut.equity_futures_px_to_db(es_prop_roll_futs_final, roll_type='proportional')

        es_flatpx_roll_futs_final = cont_fut.run_equity_futures_roll("ES", "CHRIS/CME_ES1", "CHRIS/CME_ES2",
                                                                     "2017-01-01",
                                                                     "2021-12-31", [2017, 2018, 2019, 2020], 4,
                                                                     roll_type='flatprice')
        cont_fut.write_to_csv(es_flatpx_roll_futs_final, "ES_FlatPrice_Roll", "2017-01-01", "2021-12-31", force_new_file=True)
        cont_fut.equity_futures_px_to_db(es_flatpx_roll_futs_final, roll_type='flatprice')
        # NQ Futures
        nq_prop_roll_futs_final = cont_fut.run_equity_futures_roll("NQ", "CHRIS/CME_NQ1", "CHRIS/CME_NQ2", "2017-01-01",
                                                                   "2020-12-31", [2017, 2018, 2019, 2020], 16,
                                                                   roll_type='proportional')
        cont_fut.write_to_csv(nq_prop_roll_futs_final, "NQ_Proportional_Roll", "2017-01-01", "2020-12-31", force_new_file=True)
        cont_fut.equity_futures_px_to_db(nq_prop_roll_futs_final, roll_type='proportional')
        nq_flatpx_roll_futs_final = cont_fut.run_equity_futures_roll("NQ", "CHRIS/CME_NQ1", "CHRIS/CME_NQ2",
                                                                     "2017-01-01",
                                                                     "2020-12-31", [2017, 2018, 2019, 2020], 16,
                                                                     roll_type='flatprice')
        cont_fut.write_to_csv(nq_flatpx_roll_futs_final, "NQ_FlatPrice_Roll", "2017-01-01", "2020-12-31", force_new_file=True)
        cont_fut.equity_futures_px_to_db(nq_flatpx_roll_futs_final, roll_type='flatprice')
        # YM Futures (Dow 30)
        ym_prop_roll_futs_final = cont_fut.run_equity_futures_roll("YM", "CHRIS/CME_YM1", "CHRIS/CME_YM2", "2017-01-01",
                                                                   "2020-12-31", [2017, 2018, 2019, 2020], 16,
                                                                   roll_type='proportional')
        cont_fut.write_to_csv(ym_prop_roll_futs_final, "YM_Proportional_Roll", "2017-01-01", "2020-12-31", force_new_file=True)
        cont_fut.equity_futures_px_to_db(ym_prop_roll_futs_final, roll_type='proportional')
        ym_flatpx_roll_futs_final = cont_fut.run_equity_futures_roll("YM", "CHRIS/CME_YM1", "CHRIS/CME_YM2",
                                                                     "2017-01-01",
                                                                     "2020-12-31", [2017, 2018, 2019, 2020], 16,
                                                                     roll_type='flatprice')
        cont_fut.write_to_csv(ym_flatpx_roll_futs_final, "YM_FlatPrice_Roll", "2017-01-01", "2020-12-31", force_new_file=True)
        cont_fut.equity_futures_px_to_db(ym_flatpx_roll_futs_final, roll_type='flatprice')
    elif run_type == "DAILY":
        es_prop_roll_futs_final = cont_fut.run_equity_futures_roll_daily("ES", "CHRIS/CME_ES1", "CHRIS/CME_ES2",
                                                                         "2017-01-01",
                                                                         "2021-12-31", [2017, 2018, 2019, 2020], 16,
                                                                         roll_type='proportional')[0]
        cont_fut.write_to_csv(es_prop_roll_futs_final, "ES_Proportional_Roll", "2017-01-01", "2021-12-31")
        cont_fut.equity_futures_px_to_db(es_prop_roll_futs_final, roll_type='proportional')
        es_flatpx_roll_futs_final = cont_fut.run_equity_futures_roll_daily("ES", "CHRIS/CME_ES1", "CHRIS/CME_ES2",
                                                                           "2017-01-01",
                                                                           "2021-12-31", [2017, 2018, 2019, 2020], 16,
                                                                           roll_type='flatprice')[0]
        cont_fut.write_to_csv(es_flatpx_roll_futs_final, "ES_FlatPrice_Roll", "2017-01-01", "2021-12-31")
        cont_fut.equity_futures_px_to_db(es_flatpx_roll_futs_final, roll_type='flatprice')
        # NQ Futures
        nq_prop_roll_futs_final = cont_fut.run_equity_futures_roll_daily("NQ", "CHRIS/CME_NQ1", "CHRIS/CME_NQ2",
                                                                         "2017-01-01",
                                                                         "2020-12-31", [2017, 2018, 2019, 2020], 16,
                                                                         roll_type='proportional')[0]
        cont_fut.write_to_csv(nq_prop_roll_futs_final, "NQ_Proportional_Roll", "2017-01-01", "2020-12-31")
        cont_fut.equity_futures_px_to_db(nq_prop_roll_futs_final, roll_type='proportional')
        nq_flatpx_roll_futs_final = cont_fut.run_equity_futures_roll_daily("NQ", "CHRIS/CME_NQ1", "CHRIS/CME_NQ2",
                                                                           "2017-01-01",
                                                                           "2020-12-31", [2017, 2018, 2019, 2020], 16,
                                                                           roll_type='flatprice')[0]
        cont_fut.write_to_csv(nq_flatpx_roll_futs_final, "NQ_FlatPrice_Roll", "2017-01-01", "2020-12-31")
        cont_fut.equity_futures_px_to_db(nq_flatpx_roll_futs_final, roll_type='flatprice')
        # YM Futures (Dow 30)
        ym_prop_roll_futs_final = cont_fut.run_equity_futures_roll_daily("YM", "CHRIS/CME_YM1", "CHRIS/CME_YM2",
                                                                         "2017-01-01",
                                                                         "2020-12-31", [2017, 2018, 2019, 2020], 16,
                                                                         roll_type='proportional')[0]
        cont_fut.write_to_csv(ym_prop_roll_futs_final, "YM_Proportional_Roll", "2017-01-01", "2020-12-31")
        cont_fut.equity_futures_px_to_db(ym_prop_roll_futs_final, roll_type='proportional')
        ym_flatpx_roll_futs_final = cont_fut.run_equity_futures_roll_daily("YM", "CHRIS/CME_YM1", "CHRIS/CME_YM2",
                                                                           "2017-01-01",
                                                                           "2020-12-31", [2017, 2018, 2019, 2020], 16,
                                                                           roll_type='flatprice')[0]
        cont_fut.write_to_csv(ym_flatpx_roll_futs_final, "YM_FlatPrice_Roll", "2017-01-01", "2020-12-31")
        cont_fut.equity_futures_px_to_db(ym_flatpx_roll_futs_final, roll_type='flatprice')


def run_cme_daily_ftp():
    """
    Right now, all this does is pull the XML/CSV Zip files for settlement
    We are not using the data yet.
    :return:
    """
    run_time = time(17, 0, 0, 0)
    if datetime.now().time() > run_time and datetime.now().date().weekday() < 5:
        # we can run this with today date
        rd_dt = datetime.now().date()
    else:
        if datetime.now().date().weekday() == 6:
            rd_dt = datetime.now().date() - timedelta(days=2)
        elif datetime.now().date().weekday() == 0:
            rd_dt = datetime.now().date() - timedelta(days=3)
        else:
            rd_dt = datetime.now().date() - timedelta(days=1)
    dcsr = cont_fut.DailyCMESettlementReport(report_date_dt=rd_dt)
    dcsr.get_all_reports()


def run_cboe_daily():
    cboe.run_cboe_eod()


def run_nominal_yield_curve():
    nominal_yield_curve.run_month_insert()


def run_real_yield_curve():
    real_yield_curve.run_month_insert()


# run_index_etf(run_type="HISTORICAL")
run_cboe_daily()
run_currencies(run_type="DAILY")
run_vix_futs_curve(run_type="DAILY")
run_crude_oil(run_type="DAILY")
run_eurodollars(run_type="DAILY")
run_equity_futures(run_type="DAILY")
run_cme_daily_ftp()
run_nominal_yield_curve()
run_real_yield_curve()