'''
Created on Nov 21, 2017

@author: traderghazy
'''

import requests
from data_object import DataObject
#from root.nested.data_object import DataObject
from os_mux import OSMuxImpl
#from root.nested.os_mux import OSMuxImpl
from root.nested import get_logger

class QuandlSymbolInterface(object):
    
    def __init__(self,
                 new_symbols_dict=None):
        
        self.logger = get_logger()
        
        self.quandl_FOREX_symbols_dict = {'BRZ_REAL_USD_spot' : 'BOE/XUDLB8KL',
                                          'YUAN_USD_spot' : 'BOE/XUDLBK73',
                                          'CZK_KORUNA_STERLING_spot' : 'BOE/XUDLBK44',
                                          'DANISH_KRONE_USD_spot' : 'BOE/XUDLDKD',
                                          'TKY_LIRA_STERLING_spot' : 'BOE/XUDLBK95',
                                          'SHEKL_STERLING_spot' : 'BOE/XUDLBK78',
                                          'PLD_ZLOTY_EURO_spot' : 'BOE/XUDLBK48',
                                          'DANISH_KRONE_STERLING_spot' : 'BOE/XUDLDKS',
                                          'RUBLE_USD_spot' : 'BOE/XUDLBK69',
                                          'TKY_LIRA_USD_spot' : 'BOE/XUDLBK75',
                                          'HGY_FORNT_USD_spot' : 'BOE/XUDLBK35',
                                          'SHEKL_USD_spot' : 'BOE/XUDLBK65',
                                          'YUAN_STERLING_spot' : 'BOE/XUDLBK89',
                                          'CAD_STERLING_spot' : 'BOE/XUDLCDS',
                                          'IND_RUPEE_STERLING_spot' : 'BOE/XUDLBK97',
                                          'STERLING_EURO_spot' : 'BOE/XUDLSER',
                                          'CZK_KORUNA_EURO_spot' : 'BOE/XUDLBK26',
                                          'MLY_RINGGIT_USD_spot' : 'BOE/XUDLBK66',
                                          'SWD_KRONA_STERLING_spot' : 'BOE/XUDLSKS',
                                          'YEN_EURO_spot' : 'BOE/XUDLBK63',
                                          'EURO_STERLING_spot' : 'BOE/XUDLERS',
                                          'AUD_USD_spot' : 'BOE/XUDLADD',
                                          'EURO_USD_spot' : 'BOE/XUDLERD',
                                          'STERLING_USD_spot': 'BOE/XUDLGBD',
                                          'CAD_USD_spot' : 'BOE/XUDLCDD',
                                          'SUI_FRANC_STERLING_spot' : 'BOE/XUDLSFS',
                                          'RUBLE_STERLING_spot' : 'BOE/XUDLBK85',
                                          'NZ_DOLLAR_STERLING_spot' : 'BOE/XUDLNDS',
                                          'SGD_STERLING_spot' : 'BOE/XUDLSGS',
                                          'AUD_STERLING_spot' : 'BOE/XUDLADS',
                                          'DANISH_KRONE_EURO_spot' : 'BOE/XUDLBK76',
                                          'SK_WON_STERLING_spot' : 'BOE/XUDLBK93',
                                          'PLD_ZLOTY_STERLING_spot' : 'BOE/XUDLBK47',
                                          'SGD_USD_spot': 'BOE/XUDLSGD'
                                          }
        
        self.quandl_INTEREST_RATE_symbols_dict = {'US_TGT_FED_FUNDS(UPPER_LIMIT)':'FRED/DFEDTARU',
                                                  'US_EFFECTIVE_FED_FUNDS':'FRED/DFF',
                                                  'US_TREASURY_YIELD_CURVE':'USTREASURY/YIELD'
                                                  }
        
        self.quandl_FED_FORECAST_symbols_dict = {'US_10YR_TRY_BOND_MEDIAN_VALUES_FORECAST': 'FRBP/TBOND_MD'
                                                 }
        
        self.quandl_ECONOMIC_INDICATORS_SEAS_ADJ_symbols_dict = {'US_CPI_U_ALL_SEAS_ADJ': 'FRED/CPIAUCSL',
                                                'US_CPI_U_ALL_MINUS_FOOD_ENERGY_SEAS_ADJ': 'FRED/CPILFESL',
                                                'US_CPI_U_HOUSING_SEAS_ADJ': 'BLSI/CUSR0000SAH',
                                                'US_CPI_U_COMMODITIES_SEAS_ADJ': 'BLSI/CUSR0000SAC',
                                                'US_CPI_U_ENERGY_SEAS_ADJ': 'BLSI/CUSR0000SA0E',
                                                'US_CPI_U_ELECTRICITY_SEAS_ADJ': 'BLSI/CUSR0000SEHF01',
                                                'US_CPI_U_FOOD_BEVERAGES_SEAS_ADJ': 'BLSI/CUSR0000SAF',
                                                'US_CPI_U_PUBLIC_TRANS_SEAS_ADJ': 'BLSI/CUSR0000SETG',
                                                'US_CPI_U_AIRLINE_FARE_SEAS_ADJ': 'BLSI/CUSR0000SETG01',
                                                'US_CPI_U_GASOLINE_SEAS_ADJ': 'BLSI/CUSR0000SETB01',
                                                'US_CPI_U_UTILITIES_PUBLIC_TRANS_SEAS_ADJ': 'BLSI/CUSR0000SAS24',
                                                'US_CPI_U_MEDIAN_PCT_CHANGE_SEAS_ADJ':'MEDCPIM157SFRBCLE',
                                                'US_CPI_U_MEDIAN_ANNUALIZED_PCT_CHANGE_SEAS_ADJ':'FRED/MEDCPIM158SFRBCLE',
                                                'US_CPI_U_MEDIAN_Y_OVER_Y_PCT_CHANGE_SEAS_ADJ':'FRED/MEDCPIM157SFRBCLE',
                                                'US_PCE_ANNUAL_RATE_SEAS_ADJ':'FRED/PCE',
                                                'US_REAL_PCE_ANNUAL_RATE_SEAS_ADJ':'FRED/PCEC96',
                                                'US_PCE_CORE_MINUS_FOOD_ENERGY_SEAS_ADJ': 'FRED/PCEPILFE',
                                                'US_REAL_DPI_ANNUAL_RATE_SEAS_ADJ':'FRED/DSPIC96',
                                                'US_PPI_FD_SEAS_ADJ': 'FRED/PPIFIS',
                                                'US_PPI_FD_GOODS_SEAS_ADJ': 'FRED/PPIDGS',
                                                'US_PPI_FD_ENERGY_SEAS_ADJ': 'FRED/PPIFDE',
                                                'US_PPI_FD_LESS_FOODS_ENERGY_SEAS_ADJ': 'FRED/PPIFES',
                                                'US_PPI_FD_SERVICES_SEAS_ADJ': 'FRED/PPIDSS',
                                                'US_PPI_FD_CONSTR_SEAS_ADJ': 'FRED/PPIDCS',
                                                'US_PPI_FD_FINISHED_GOODS_SEAS_ADJ': 'FRED/WPSFD49207',
                                                'US_PPI_FD_FINISHED_CORE_SEAS_ADJ': 'FRED/WPSFD4131',
                                                'US_EMPLOYMENT_SEAS_ADJ': 'FRED/PAYEMS',
                                                'US_AVG_WEEKLY_HOURS_SEAS_ADJ': 'FRED/AWHAETP',
                                                'US_AVG_WEEKLY_EARNINGS_SEAS_ADJ':'FRED/CES0500000003',
                                                'US_WEEKLY_JOBLESS_CLAIMS_SEAS_ADJ':'FRED/ICSA',
                                                'US_REAL_GDP_ANNUAL_RATE_SEAS_ADJ':'FRED/GDPC1',
                                                'US_REAL_GDP_PCT_CHANGE_ANNUAL_RATE_SEAS_ADJ':'FRED/A191RL1Q225SBEA',
                                                'US_ECOMMERCE_RETAIL_SALES_PCT_TOTAL_SALES_QRTLY_SEAS_ADJ':'FRED/ECOMPCTSA',
                                                'US_RETAIL_SALES_RETAIL_EXCLUDING_FOOD_SERVICES_SEAS_ADJ':'FRED/FSXFS',
                                                'US_RETAIL_ECOMMERCE_SALES_SEAS_ADJ':'FRED/ECOMSA',
                                                'US_RETAIL_RETAILER_SALES_SEAS_ADJ':'FRED/RETAILSMSA',
                                                'US_RETAIL_RETAILERS_INVENTORIES_TO_SALES_RATIOS_SEAS_ADJ':'FRED/RETAILIRSA',
                                                'US_RETAIL_SALES_RETAIL_FOOD_SERVES_EXCLUDING_MOTOR_VEHICLES_PARTS_SEAS_ADJ':'FRED/RSFSXMV',
                                                'US_RETAIL_SALES_CLOTHING_SEAS_ADJ':'FRED/RSCCAS',
                                                'US_RETAIL_SALES_FURNITURE_SEAS_ADJ':'FRED/RSFHFS',
                                                'US_RETAIL_SALES_DEPARTMENT_STORES_SEAS_ADJ':'FRED/RSDSELD',
                                                'US_RETAIL_SALES_MOTOR_VEHICLE_DEALERS_SEAS_ADJ':'FRED/RSMVPD',
                                                'US_RETAIL_SALES_ELECTRONICS_APPLIANCES_STORES_SEAS_ADJ':'FRED/RESEAS',
                                                'US_RETAIL_SALES_GASOLINE_STATIONS_SEAS_ADJ':'FRED/RSGASS',
                                                'US_RETAIL_SALES_NON_STORE_RETAILERS_SEAS_ADJ':'FRED/RSNSR',
                                                'US_RETAIL_SALES_SPORTING_GOODS_HOBBY_BOOK_MUSIC_STORES_SEAS_ADJ':'FRED/RSSGHBMS',
                                                'US_IP_SEAS_ADJ':'FRED/INDPRO',
                                                'US_IP_MANUFACTURING(NAICS)_SEAS_ADJ':'FRED/IPMAN',
                                                'US_IP_MINING_CRUDE_OIL_SEAS_ADJ':'FRED/IPG211111CS',
                                                'US_IP_DURABLE_CONSUMER_GOODS_SEAS_ADJ':'FRED/IPDCONGD',
                                                'US_IP_ELECTRIC_GAS_UTIL_SEAS_ADJ':'FRED/IPUTIL',
                                                'US_IP_CONSUMER_GOODS_SEAS_ADJ':'FRED/IPCONGD',
                                                'US_ISM_MFG_PMI_SEAS_ADJ':'ISM/MAN_PMI',
                                                'US_ISM_MFG_PROD_INDEX_SEAS_ADJ':'ISM/MAN_PROD',  ## Diffusion_Index is the column
                                                'US_ISM_NON_MFG_INDEX_SEAS_ADJ':'ISM/NONMAN_NMI',
                                                'US_ISM_NON_MFG_PRICES_INDEX_SEAS_ADJ':'ISM/NONMAN_PRICES', ##Diffusion_Index is the column
                                                'US_PHILY_FED_BUS_OUTLOOK_CURR_ACTIVITY_INDEX_SEAS_ADJ':'FRBP/GAC',
                                                'US_PHILY_FED_BUS_OUTLOOK_CURR_NEW_ORDERS_SEAS_ADJ':'FRBP/NOC',
                                                'US_PHILY_FED_BUS_OUTLOOK_FUT_NEW_ORDERS_SEAS_ADJ':'FRBP/NOF',
                                                'US_PHILY_FED_BUS_OUTLOOK_CURR_SHIPMENTS_SEAS_ADJ':'FRBP/SHC',
                                                'US_PHILY_FED_BUS_OUTLOOK_FUT_SHIPMENTS_SEAS_ADJ':'FRBP/SHF',
                                                'US_PHILY_FED_BUS_OUTLOOK_CURR_UNFILLED_ORDERS_SEAS_ADJ':'FRBP/UOC',
                                                'US_PHILY_FED_BUS_OUTLOOK_CURR_INVENTORIES_SEAS_ADJ':'FRBP/IVC',
                                                'US_PHILY_FED_BUS_OUTLOOK_FUT_INVENTORIES_SEAS_ADJ':'FRBP/IVF',
                                                'US_PHILY_FED_BUS_OUTLOOK_CURR_PRICES_PAID_SEAS_ADJ':'FRBP/PPC',
                                                'US_PHILY_FED_BUS_OUTLOOK_CURR_EMPLOYMENT_SEAS_ADJ':'FRBP/NEC',
                                                'US_PHILY_FED_BUS_OUTLOOK_CURR_AVG_WORKWEEK_SEAS_ADJ':'FRBP/AWC',
                                                'US_PHILY_FED_BUS_OUTLOOK_FUT_AVG_WORKWEEK_SEAS_ADJ':'FRBP/AWF',
                                                'US_PHILY_FED_BUS_OUTLOOK_FUT_ACTIVITY_INDEX_SEAS_ADJ':'FRBP/GAF',
                                                'US_PHILY_FED_BUS_OUTLOOK_FUT_EMPLOYMENT_SEAS_ADJ':'FRBP/NEF',
                                                'US_ARUOBA_DIEBOLD_SCOTTI_BUS_CORREL_INDEX_SEAS_ADJ':'FRBP/ADS_VINTAGES_MOSTRECENT'
                                                }
        
        self.quandl_ECONOMIC_INDICATORS_UNADJ_symbols_dict = {'US_CPI_U_ALL_UNADJ': 'FRED/CPIAUCNS',
                                                              'US_CPI_U_ENERGY_UNADJ': 'BLSI/CUUR0000SA0E',
                                                              'US_CPI_U_TRANS_UNADJ': 'BLSI/CUUR0000SAT',
                                                              'US_CPI_U_ALL_UNADJ': 'BLSI/CUUR0000SA0',
                                                              'US_CPI_U_FUES_UTILS_UNADJ': 'BLSI/CUUR0000SAH2',
                                                              'US_CHAINED_CPI_U_ALL_UNADJ': 'BLSI/SUUR0000SA0',
                                                              'US_PPI_UTILITIES': 'BLSI/PCU221_221',
                                                              'US_PPI_MINING_MINUS_OILGAS': 'BLSI/PCU212_212',
                                                              'US_PPI_COAL_MINING': 'FRED/PCU21212121',
                                                              'US_PPI_OIL_GAS_EXTRACTION': 'BLSI/PCU211_211',
                                                              'US_PPI_POWER_GEN_TRANS_DIST': 'BLSI/PCU2211_2211',
                                                              'US_PPI_GOLD_SILVER_MINING': 'FRED/PCU2122221222',
                                                              'US_PPI_NAT_GAS_DIST': 'FRED/PCU22122212',
                                                              'US_PPI_CHEMICAL_MFG': 'FRED/PCU325325',
                                                              'US_PPI_TRUCK_TRANS': 'FRED/PCU484484',
                                                              'US_PPI_IRON_STEEL_MILLS': 'FRED/PCU331110331110',
                                                              'US_PPI_GENERAL_FREIGHT_TRUCKING': 'FRED/PCU48414841',
                                                              'US_PPI_FD_UNADJ': 'FRED/PPIFID',
                                                              'US_PPI_FD_GOODS_UNADJ': 'FRED/PPIFDG',
                                                              'US_PPI_FD_ENERGY_UNADJ': 'FRED/PPIFDE',
                                                              'US_PPI_FD_LESS_FOODS_ENERGY_UNADJ': 'FRED/PPICOR',
                                                              'US_PPI_FD_SERVICES_UNADJ': 'FRED/PPIFDS',
                                                              'US_PPI_FD_CONSTR_UNADJ': 'FRED/PPIFDC',
                                                              'US_PPI_GASOLINE': 'FRED/WPS0571',
                                                              'US_PPI_INDUSTRIAL_CHEMICALS': 'FRED/WPU061',
                                                              'US_PPI_LUMBER': 'FRED/WPU081',
                                                              'US_PPI_DIESEL_FUEL': 'FRED/WPU057303',
                                                              'US_PPI_CRUDE_PETROL': 'FRED/WPU0561',
                                                              'US_PPI_TRUCK_TRANS_FREIGHT': 'FRED/WPU3012',
                                                              'US_PPI_FD_FINISHED_GOODS_UNADJ': 'FRED/WPUFD49207',
                                                              'US_PPI_FD_FINISHED_CORE_UNADJ': 'FRED/WPUFD4131',
                                                              'US_PPI_ALL_COMMODITIES_UNADJ': 'FRED/PPIACO',
                                                              'US_IP_MINING_CRUDE_OIL_UNADJ':'FRED/IPG211111CN',
                                                              'US_ISM_MFG_PRICES_INDEX_UNADJ':'ISM/MAN_PRICES',
                                                              }
        
        self.quandl_FOREX_symbols_dict_reversed = {'BOE/XUDLERD' : 'EURO_USD_spot', 
                                                   'BOE/XUDLSER' : 'STERLING_EURO_spot',
                                                   'BOE/XUDLERS' : 'EURO_STERLING_spot',
                                                   'BOE/XUDLB8KL' : 'BRZ_REAL_USD_spot',
                                                   'BOE/XUDLBK95' : 'TKY_LIRA_STERLING_spot',
                                                   'BOE/XUDLBK89' : 'CHN_YUAN_STERLING_spot',
                                                   'BOE/XUDLBK97' : 'IND_RUPEE_STERLING_spot',
                                                   'BOE/XUDLBK93' : 'SK_WON_STERLING_spot',
                                                   'BOE/XUDLDKD' : 'DANISH_KRONE_USD_spot',
                                                   'BOE/XUDLDKS' : 'DANISH_KRONE_STERLING_spot',
                                                   'BOE/XUDLBK76' : 'DANISH_KRONE_EURO_spot',
                                                   'BOE/XUDLSGS' : 'SGD_STERLING_spot',
                                                   'BOE/XUDLSKS' : 'SWD_KRONA_STERLING_spot',
                                                   'BOE/XUDLBK48' : 'PLD_ZLOTY_EURO_spot',
                                                   'BOE/XUDLSFS' : 'SUI_FRANC_STERLING_spot',
                                                   'BOE/XUDLNDS' : 'NZ_DOLLAR_STERLING_spot',
                                                   'BOE/XUDLBK47' : 'PLD_ZLOTY_STERLING_spot',
                                                   'BOE/XUDLADS' : 'AUD_STERLING_spot',
                                                   'BOE/XUDLCDS' : 'CAD_STERLING_spot',
                                                   'BOE/XUDLCDD' : 'CAD_USD_spot',
                                                   'BOE/XUDLADD' : 'AUD_USD_spot',
                                                   'BOE/XUDLSGD' : 'SGD_USD_spot',
                                                   'BOE/XUDLBK75' : 'TKY_LIRA_USD_spot',
                                                   'BOE/XUDLBK66' : 'MLY_RINGGIT_USD_spot',
                                                   'BOE/XUDLBK35' : 'HGY_FORNT_USD_spot',
                                                   'BOE/XUDLBK73' : 'YUAN_USD_spot',
                                                   'BOE/XUDLBK69' : 'RUBLE_USD_spot',
                                                   'BOE/XUDLBK65' : 'SHEKL_USD_spot',
                                                   'BOE/XUDLBK63' : 'YEN_EURO_spot',
                                                   'BOE/XUDLBK85' : 'RUBLE_STERLING_spot',
                                                   'BOE/XUDLBK89' : 'YUAN_STERLING_spot',
                                                   'BOE/XUDLGBD' : 'STERLING_USD_spot',
                                                   'BOE/XUDLBK44' : 'CZK_KORUNA_STERLING_spot',
                                                   'BOE/XUDLBK26' : 'CZK_KORUNA_EURO_spot',
                                                   'BOE/XUDLBK78' : 'SHEKL_STERLING_spot'
                                                   }
        
        self.quandl_EURODOLLARS_symbols_dict = {'ED1_WHITE_CONTINUOUS_CONTRACT': 'CHRIS/CME_ED1',
                                                     'ED2_WHITE_CONTINUOUS_CONTRACT': 'CHRIS/CME_ED2',
                                                     'ED3_WHITE_CONTINUOUS_CONTRACT': 'CHRIS/CME_ED3',
                                                     'ED4_WHITE_CONTINUOUS_CONTRACT': 'CHRIS/CME_ED4',
                                                     'ED5_RED_CONTINUOUS_CONTRACT': 'CHRIS/CME_ED5',
                                                     'ED6_RED_CONTINUOUS_CONTRACT': 'CHRIS/CME_ED6',
                                                     'ED7_RED_CONTINUOUS_CONTRACT': 'CHRIS/CME_ED7',
                                                     'ED8_RED_CONTINUOUS_CONTRACT': 'CHRIS/CME_ED8',
                                                     'ED9_GREEN_CONTINUOUS_CONTRACT': 'CHRIS/CME_ED9',
                                                     'ED10_GREEN_CONTINUOUS_CONTRACT': 'CHRIS/CME_ED10',
                                                     'ED11_GREEN_CONTINUOUS_CONTRACT': 'CHRIS/CME_ED11',
                                                     'ED12_GREEN_CONTINUOUS_CONTRACT': 'CHRIS/CME_ED12',
                                                     'ED13_BLUE_CONTINUOUS_CONTRACT': 'CHRIS/CME_ED13',
                                                     'ED14_BLUE_CONTINUOUS_CONTRACT': 'CHRIS/CME_ED14',
                                                     'ED15_BLUE_CONTINUOUS_CONTRACT': 'CHRIS/CME_ED15',
                                                     'ED16_BLUE_CONTINUOUS_CONTRACT': 'CHRIS/CME_ED16',
                                                     'ED17_GOLD_CONTINUOUS_CONTRACT': 'CHRIS/CME_ED17',
                                                     'ED18_GOLD_CONTINUOUS_CONTRACT': 'CHRIS/CME_ED18',
                                                     'ED19_GOLD_CONTINUOUS_CONTRACT': 'CHRIS/CME_ED19',
                                                     'ED20_GOLD_CONTINUOUS_CONTRACT': 'CHRIS/CME_ED20',
                                                     'ED21_PURPLE_CONTINUOUS_CONTRACT': 'CHRIS/CME_ED21',
                                                     'ED22_PURPLE_CONTINUOUS_CONTRACT': 'CHRIS/CME_ED22',
                                                     'ED23_PURPLE_CONTINUOUS_CONTRACT': 'CHRIS/CME_ED23',
                                                     'ED24_PURPLE_CONTINUOUS_CONTRACT': 'CHRIS/CME_ED24'}
        
        self.new_symbols_dict = new_symbols_dict
        self.local_quandl_forex_data_path = '/workspace/data/quandl/forex/'
        self.local_quandl_economic_indicators_data_path = '/workspace/data/quandl/economic_indcators/'
        self.local_quandl_interest_rate_data_path = '/workspace/data/quandl/interest_rates/'
        self.local_quandl_fed_forecasts_data_path = '/workspace/data/quandl/fed_forecasts/'
        self.local_yahoo_data_path = '/workspace/data/yahoo/'
        self.local_misc_data_path = '/workspace/data/'
        self.local_stock_data_path = '/workspace/data/quandl/stocks/'
        
        self.quandl_auth_token="vGeP2BmzAigd5D1PRDk_"
    
        self.quandl_data_class_mapper = {'FOREX':[self.quandl_FOREX_symbols_dict,self.local_quandl_forex_data_path],
                                         'EURODOLLARS':[self.quandl_EURODOLLARS_symbols_dict, self.local_quandl_interest_rate_data_path],
                                         'INTEREST_RATES':[self.quandl_INTEREST_RATE_symbols_dict,self.local_quandl_interest_rate_data_path],
                                         'ECONOMIC_INDICATORS_UNADJ':[self.quandl_ECONOMIC_INDICATORS_UNADJ_symbols_dict, self.local_quandl_economic_indicators_data_path],
                                         'ECONOMIC_INDICATORS_SEAS_ADJ':[self.quandl_ECONOMIC_INDICATORS_SEAS_ADJ_symbols_dict, self.local_quandl_economic_indicators_data_path],
                                         'FED_FORECASTS':[self.quandl_FED_FORECAST_symbols_dict, self.local_quandl_fed_forecasts_data_path],
                                         'MISC': [self.local_misc_data_path],
                                         'STOCKS': [self.local_stock_data_path]}
    
    def get_quandl_symbol(self,
                          class_of_data,
                          local_symbol):
        
        return self.quandl_data_class_mapper[class_of_data][0][local_symbol]
    
    def get_local_quandl_data_path(self,
                                   class_of_data):
        
        try:
            return self.quandl_data_class_mapper[class_of_data][1]
        except:
            return self.local_misc_data_path
    
    def get_local_yahoo_data_path(self):
        
        try:
            return self.local_yahoo_data_path
        except:
            return self.local_misc_data_path
        
    def download_quandl_stock_csv_file(self,
                                       idx):
        
        url = "https://www.quandl.com/api/v3/datasets/EOD/" + idx + ".csv?api_key=" + self.quandl_auth_token
        response = requests.get(url)
        local_data_file_pwd = OSMuxImpl.get_proper_path(self.local_stock_data_path)
        total_local_file_name = local_data_file_pwd + idx + ".csv"
        ### write out the response to file
        with open(total_local_file_name, 'wb') as f:
            f.write(response.content)
        self.logger.info("QuandlSymbolInterface.download_quandl_stock_csv_file(): HTTP Response Status Code %s " + \
                         str(response.status_code))
        
    def daily_batch_equity_download(self):
        
        do = DataObject()
        suf_df = do.get_stock_universe_file_as_df()
        index_np_array = suf_df.index
        #do_dwnld = lambda idx : self.download_quandl_stock_csv_file(idx)
        for idx in index_np_array:
            self.download_quandl_stock_csv_file(idx)
        
    def daily_batch_update_quandl_symbols(self):
        ## update the csv files with new data from daily update.
        ## what time to do this? Need to find out when Quandl updates their data
        return 1

    def single_symbol_update(self,
                             local_symbol):
        # this function accepts a non-Quandl symbol and updates the local csv data file
        return 1
    