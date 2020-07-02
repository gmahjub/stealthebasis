'''
Created on Nov 21, 2017

@author: traderghazy
'''

import requests
from root.nested.dataAccess.data_object import DataObject
from root.nested.SysOs.os_mux import OSMuxImpl
from root.nested import get_logger


class QuandlSymbolInterface(object):

    def __init__(self,
                 new_symbols_dict=None):

        self.logger = get_logger()

        self.wilshire_tr_index_dict = {'Wilshire US Large-Cap Total Market Index': 'WILLLRGCAP'}
        self.baml_tr_fi_index_dict = {
            'ICE BofAML US High Yield Master II Total Return Index Value': 'BAMLHYH0A0HYM2TRIV',
        }
        # reference: https://fred.stlouisfed.org/categories/32413
        self.baml_tr_fi_category_id = 32413
        # reference: https: // fred.stlouisfed.org / categories / 32255
        self.stock_market_indexes = 32255

        self.to_usd_sym_dict = {'AUD_USD_spot': 'BOE/XUDLADD',  # aussie dollar
                                'CAD_USD_spot': 'BOE/XUDLCDD',  # canadian dollar
                                'CNY_USD_spot': 'BOE/XUDLBK73',  # yuan (China)
                                'DKK_USD_spot': 'BOE/XUDLDKD',  # danish krone
                                'HKD_USD_spot': 'BOE/XUDLHDD',  # hong kong dollar
                                'HUF_USD_spot': 'BOE/XUDLBK35',  # hungarian forint
                                'INR_USD_spot': 'BOE/XUDLBK64',  # indian rupee
                                'JPY_USD_spot': 'BOE/XUDLJYD',  # jp yen
                                'MYR_USD_spot': 'BOE/XUDLBK66',  # malaysian ringgit
                                'NZD_USD_spot': 'BOE/XUDLNDD',  # new zealand dollar
                                'NOK_USD_spot': 'BOE/XUDLNKD',  # norwegian krone
                                'PLN_USD_spot': 'BOE/XUDLBK49',  # polish zloty
                                'GBP_USD_spot': 'BOE/XUDLGBD',  # british pound
                                'RUB_USD_spot': 'BOE/XUDLBK69',  # russian ruble
                                'SAR_USD_spot': 'BOE/XUDLSRD',  # saudi riyal
                                'SGD_USD_spot': 'BOE/XUDLSGD',  # singapore dollar
                                'ZAR_USD_spot': 'BOE/XUDLZRD',  # south african rand
                                'KRW_USD_spot': 'BOE/XUDLBK74',  # south korean won
                                'SEK_USD_spot': 'BOE/XUDLSKD',  # swedish krona
                                'CHF_USD_spot': 'BOE/XUDLSFD',  # swiss franc
                                'TRY_USD_spot': 'BOE/XUDLBK75',  # turkish lira,
                                'BRZ_USD_spot': 'BOE/XUDLB8KL',  # brazilian real
                                'NIS_USD_spot': 'BOE/XUDLBK65',  # israeli shekyl
                                'CZK_USD_spot': 'BOE/XUDLBK27',  # czech koruna
                                'EUR_USD_spot': 'BOE/XUDLERD'  # euro
                                }

        self.to_eur_sym_dict = {'PLN_EUR_spot': 'BOE/XUDLBK48',
                                'GBP_EUR_spot': 'BOE/XUDLSER',
                                'CZK_EUR_spot': 'BOE/XUDLBK26',
                                'JPY_EUR_spot': 'BOE/XUDLBK63',
                                'DKK_EUR_spot': 'BOE/XUDLBK76'
                                }

        self.to_gbp_sym_dict = {'TRY_GBP_spot': 'BOE/XUDLBK95',  # turkish lira
                                'NIS_GBP_spot': 'BOE/XUDLBK78',  # israeli shekyl
                                'DKK_GBP_spot': 'BOE/XUDLDKS',  # danish krone
                                'CNY_GBP_spot': 'BOE/XUDLBK89',  # china yuan
                                'CAD_GBP_spot': 'BOE/XUDLCDS',  # canadian dollar
                                'INR_GBP_spot': 'BOE/XUDLBK97',  # indian rupee
                                'SEK_GBP_spot': 'BOE/XUDLSKS',  # swedish krona
                                'EUR_GBP_spot': 'BOE/XUDLERS',  # euro
                                'CHF_GBP_spot': 'BOE/XUDLSFS',  # swiss franc
                                'RUB_GBP_spot': 'BOE/XUDLBK85',  # russian ruble
                                'NZD_GBP_spot': 'BOE/XUDLNDS',  # new zealand dollar
                                'SGD_GBP_spot': 'BOE/XUDLSGS',  # singapore dollar
                                'AUD_GBP_spot': 'BOE/XUDLADS',  # australian dollar
                                'KRW_GBP_spot': 'BOE/XUDLBK93',  # south korean won
                                'PLN_GBP_spot': 'BOE/XUDLBK47'  # polish zloty
                                }

        self.gdp_sym_dict = {'US_QUARTERLY_REAL_GDP_VALUE_SEAS_ADJ': 'FRED/GDPC1',
                             'US_QUARTERLY_REAL_GDP_ANNUAL_RATE_SEAS_ADJ': 'FRED/A191RL1Q225SBEA'
                             }

        self.quandl_INTEREST_RATE_symbols_dict = {'US_TGT_FED_FUNDS(UPPER_LIMIT)': 'FRED/DFEDTARU',
                                                  'US_EFFECTIVE_FED_FUNDS': 'FRED/DFF',
                                                  'US_TREASURY_YIELD_CURVE': 'USTREASURY/YIELD'
                                                  }

        self.quandl_FED_FORECAST_symbols_dict = {'US_10YR_TRY_BOND_MEDIAN_VALUES_FORECAST': 'FRBP/TBOND_MD'
                                                 }

        self.cpi_deflator_sym_dict = {'US_CPI_U_ALL_SEAS_ADJ': 'FRED/CPIAUCSL',
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
                                      'US_CPI_U_MEDIAN_PCT_CHANGE_SEAS_ADJ': 'MEDCPIM157SFRBCLE',
                                      'US_CPI_U_MEDIAN_ANNUALIZED_PCT_CHANGE_SEAS_ADJ': 'FRED/MEDCPIM158SFRBCLE',
                                      'US_CPI_U_MEDIAN_Y_OVER_Y_PCT_CHANGE_SEAS_ADJ': 'FRED/MEDCPIM157SFRBCLE'
                                      }

        self.pce_deflator_sym_dict = {'US_PCE_ANNUAL_RATE_SEAS_ADJ': 'FRED/PCE',
                                      'US_REAL_PCE_ANNUAL_RATE_SEAS_ADJ': 'FRED/PCEC96',
                                      'US_PCE_CORE_MINUS_FOOD_ENERGY_SEAS_ADJ': 'FRED/PCEPILFE'
                                      }

        self.quandl_ECONOMIC_INDICATORS_SEAS_ADJ_symbols_dict = {
            'US_REAL_DPI_ANNUAL_RATE_SEAS_ADJ': 'FRED/DSPIC96',
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
            'US_AVG_WEEKLY_EARNINGS_SEAS_ADJ': 'FRED/CES0500000003',
            'US_WEEKLY_JOBLESS_CLAIMS_SEAS_ADJ': 'FRED/ICSA',
            'US_ECOMMERCE_RETAIL_SALES_PCT_TOTAL_SALES_QRTLY_SEAS_ADJ': 'FRED/ECOMPCTSA',
            'US_RETAIL_SALES_RETAIL_EXCLUDING_FOOD_SERVICES_SEAS_ADJ': 'FRED/FSXFS',
            'US_RETAIL_ECOMMERCE_SALES_SEAS_ADJ': 'FRED/ECOMSA',
            'US_RETAIL_RETAILER_SALES_SEAS_ADJ': 'FRED/RETAILSMSA',
            'US_RETAIL_RETAILERS_INVENTORIES_TO_SALES_RATIOS_SEAS_ADJ': 'FRED/RETAILIRSA',
            'US_RETAIL_SALES_RETAIL_FOOD_SERVES_EXCLUDING_MOTOR_VEHICLES_PARTS_SEAS_ADJ': 'FRED/RSFSXMV',
            'US_RETAIL_SALES_CLOTHING_SEAS_ADJ': 'FRED/RSCCAS',
            'US_RETAIL_SALES_FURNITURE_SEAS_ADJ': 'FRED/RSFHFS',
            'US_RETAIL_SALES_DEPARTMENT_STORES_SEAS_ADJ': 'FRED/RSDSELD',
            'US_RETAIL_SALES_MOTOR_VEHICLE_DEALERS_SEAS_ADJ': 'FRED/RSMVPD',
            'US_RETAIL_SALES_ELECTRONICS_APPLIANCES_STORES_SEAS_ADJ': 'FRED/RESEAS',
            'US_RETAIL_SALES_GASOLINE_STATIONS_SEAS_ADJ': 'FRED/RSGASS',
            'US_RETAIL_SALES_NON_STORE_RETAILERS_SEAS_ADJ': 'FRED/RSNSR',
            'US_RETAIL_SALES_SPORTING_GOODS_HOBBY_BOOK_MUSIC_STORES_SEAS_ADJ': 'FRED/RSSGHBMS',
            'US_IP_SEAS_ADJ': 'FRED/INDPRO',
            'US_IP_MANUFACTURING(NAICS)_SEAS_ADJ': 'FRED/IPMAN',
            'US_IP_MINING_CRUDE_OIL_SEAS_ADJ': 'FRED/IPG211111CS',
            'US_IP_DURABLE_CONSUMER_GOODS_SEAS_ADJ': 'FRED/IPDCONGD',
            'US_IP_ELECTRIC_GAS_UTIL_SEAS_ADJ': 'FRED/IPUTIL',
            'US_IP_CONSUMER_GOODS_SEAS_ADJ': 'FRED/IPCONGD',
            'US_ISM_MFG_PMI_SEAS_ADJ': 'ISM/MAN_PMI',
            'US_ISM_MFG_PROD_INDEX_SEAS_ADJ': 'ISM/MAN_PROD',  ## Diffusion_Index is the column
            'US_ISM_NON_MFG_INDEX_SEAS_ADJ': 'ISM/NONMAN_NMI',
            'US_ISM_NON_MFG_PRICES_INDEX_SEAS_ADJ': 'ISM/NONMAN_PRICES',  ##Diffusion_Index is the column
            'US_PHILY_FED_BUS_OUTLOOK_CURR_ACTIVITY_INDEX_SEAS_ADJ': 'FRBP/GAC',
            'US_PHILY_FED_BUS_OUTLOOK_CURR_NEW_ORDERS_SEAS_ADJ': 'FRBP/NOC',
            'US_PHILY_FED_BUS_OUTLOOK_FUT_NEW_ORDERS_SEAS_ADJ': 'FRBP/NOF',
            'US_PHILY_FED_BUS_OUTLOOK_CURR_SHIPMENTS_SEAS_ADJ': 'FRBP/SHC',
            'US_PHILY_FED_BUS_OUTLOOK_FUT_SHIPMENTS_SEAS_ADJ': 'FRBP/SHF',
            'US_PHILY_FED_BUS_OUTLOOK_CURR_UNFILLED_ORDERS_SEAS_ADJ': 'FRBP/UOC',
            'US_PHILY_FED_BUS_OUTLOOK_CURR_INVENTORIES_SEAS_ADJ': 'FRBP/IVC',
            'US_PHILY_FED_BUS_OUTLOOK_FUT_INVENTORIES_SEAS_ADJ': 'FRBP/IVF',
            'US_PHILY_FED_BUS_OUTLOOK_CURR_PRICES_PAID_SEAS_ADJ': 'FRBP/PPC',
            'US_PHILY_FED_BUS_OUTLOOK_CURR_EMPLOYMENT_SEAS_ADJ': 'FRBP/NEC',
            'US_PHILY_FED_BUS_OUTLOOK_CURR_AVG_WORKWEEK_SEAS_ADJ': 'FRBP/AWC',
            'US_PHILY_FED_BUS_OUTLOOK_FUT_AVG_WORKWEEK_SEAS_ADJ': 'FRBP/AWF',
            'US_PHILY_FED_BUS_OUTLOOK_FUT_ACTIVITY_INDEX_SEAS_ADJ': 'FRBP/GAF',
            'US_PHILY_FED_BUS_OUTLOOK_FUT_EMPLOYMENT_SEAS_ADJ': 'FRBP/NEF',
            'US_ARUOBA_DIEBOLD_SCOTTI_BUS_CORREL_INDEX_SEAS_ADJ': 'FRBP/ADS_VINTAGES_MOSTRECENT'
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
                                                              'US_IP_MINING_CRUDE_OIL_UNADJ': 'FRED/IPG211111CN',
                                                              'US_ISM_MFG_PRICES_INDEX_UNADJ': 'ISM/MAN_PRICES',
                                                              }

        self.quandl_FOREX_symbols_dict_reversed = {'BOE/XUDLERD': 'EURO_USD_spot',
                                                   'BOE/XUDLSER': 'STERLING_EURO_spot',
                                                   'BOE/XUDLERS': 'EURO_STERLING_spot',
                                                   'BOE/XUDLB8KL': 'BRZ_REAL_USD_spot',
                                                   'BOE/XUDLBK95': 'TKY_LIRA_STERLING_spot',
                                                   'BOE/XUDLBK89': 'CHN_YUAN_STERLING_spot',
                                                   'BOE/XUDLBK97': 'IND_RUPEE_STERLING_spot',
                                                   'BOE/XUDLBK93': 'SK_WON_STERLING_spot',
                                                   'BOE/XUDLDKD': 'DANISH_KRONE_USD_spot',
                                                   'BOE/XUDLDKS': 'DANISH_KRONE_STERLING_spot',
                                                   'BOE/XUDLBK76': 'DANISH_KRONE_EURO_spot',
                                                   'BOE/XUDLSGS': 'SGD_STERLING_spot',
                                                   'BOE/XUDLSKS': 'SWD_KRONA_STERLING_spot',
                                                   'BOE/XUDLBK48': 'PLD_ZLOTY_EURO_spot',
                                                   'BOE/XUDLSFS': 'SUI_FRANC_STERLING_spot',
                                                   'BOE/XUDLNDS': 'NZ_DOLLAR_STERLING_spot',
                                                   'BOE/XUDLBK47': 'PLD_ZLOTY_STERLING_spot',
                                                   'BOE/XUDLADS': 'AUD_STERLING_spot',
                                                   'BOE/XUDLCDS': 'CAD_STERLING_spot',
                                                   'BOE/XUDLCDD': 'CAD_USD_spot',
                                                   'BOE/XUDLADD': 'AUD_USD_spot',
                                                   'BOE/XUDLSGD': 'SGD_USD_spot',
                                                   'BOE/XUDLBK75': 'TKY_LIRA_USD_spot',
                                                   'BOE/XUDLBK66': 'MLY_RINGGIT_USD_spot',
                                                   'BOE/XUDLBK35': 'HGY_FORNT_USD_spot',
                                                   'BOE/XUDLBK73': 'YUAN_USD_spot',
                                                   'BOE/XUDLBK69': 'RUBLE_USD_spot',
                                                   'BOE/XUDLBK65': 'SHEKL_USD_spot',
                                                   'BOE/XUDLBK63': 'YEN_EURO_spot',
                                                   'BOE/XUDLBK85': 'RUBLE_STERLING_spot',
                                                   'BOE/XUDLBK89': 'YUAN_STERLING_spot',
                                                   'BOE/XUDLGBD': 'STERLING_USD_spot',
                                                   'BOE/XUDLBK44': 'CZK_KORUNA_STERLING_spot',
                                                   'BOE/XUDLBK26': 'CZK_KORUNA_EURO_spot',
                                                   'BOE/XUDLBK78': 'SHEKL_STERLING_spot'
                                                   }

        self.quandl_EURODOLLARS_symbols_dict = {'ED1_WHITE': 'CHRIS/CME_ED1',
                                                'ED2_WHITE': 'CHRIS/CME_ED2',
                                                'ED3_WHITE': 'CHRIS/CME_ED3',
                                                'ED4_WHITE': 'CHRIS/CME_ED4',
                                                'ED5_RED': 'CHRIS/CME_ED5',
                                                'ED6_RED': 'CHRIS/CME_ED6',
                                                'ED7_RED': 'CHRIS/CME_ED7',
                                                'ED8_RED': 'CHRIS/CME_ED8',
                                                'ED9_GREEN': 'CHRIS/CME_ED9',
                                                'ED10_GREEN': 'CHRIS/CME_ED10',
                                                'ED11_GREEN': 'CHRIS/CME_ED11',
                                                'ED12_GREEN': 'CHRIS/CME_ED12',
                                                'ED13_BLUE': 'CHRIS/CME_ED13',
                                                'ED14_BLUE': 'CHRIS/CME_ED14',
                                                'ED15_BLUE': 'CHRIS/CME_ED15',
                                                'ED16_BLUE': 'CHRIS/CME_ED16',
                                                'ED17_GOLD': 'CHRIS/CME_ED17',
                                                'ED18_GOLD': 'CHRIS/CME_ED18',
                                                'ED19_GOLD': 'CHRIS/CME_ED19',
                                                'ED20_GOLD': 'CHRIS/CME_ED20',
                                                'ED21_PURPLE': 'CHRIS/CME_ED21',
                                                'ED22_PURPLE': 'CHRIS/CME_ED22',
                                                'ED23_PURPLE': 'CHRIS/CME_ED23',
                                                'ED24_PURPLE': 'CHRIS/CME_ED24'
                                                }

        self.quandl_VIX_symbols_dict = {''}

        self.new_symbols_dict = new_symbols_dict
        self.local_quandl_forex_data_path = '/workspace/data/quandl/forex/'
        self.local_quandl_economic_indicators_data_path = '/workspace/data/quandl/economic_indcators/'
        self.local_quandl_interest_rate_data_path = '/workspace/data/quandl/interest_rates/'
        self.local_quandl_fed_forecasts_data_path = '/workspace/data/quandl/fed_forecasts/'

        self.local_yahoo_data_path = '/workspace/data/yahoo/'
        self.local_misc_data_path = '/workspace/data/'
        self.local_stock_data_path = '/workspace/data/quandl/stocks/'

        self.path_to_local_gdp_data = '/workspace/data/quandl/gdp/'

        self.quandl_auth_token = "vGeP2BmzAigd5D1PRDk_"

        self.quandl_data_class_mapper = {'FOREX_TO_USD': [self.to_usd_sym_dict, self.local_quandl_forex_data_path],
                                         # 'FOREX':[self.quandl_FOREX_symbols_dict,self.local_quandl_forex_data_path],
                                         'EURODOLLARS': [self.quandl_EURODOLLARS_symbols_dict,
                                                         self.local_quandl_interest_rate_data_path],
                                         'INTEREST_RATES': [self.quandl_INTEREST_RATE_symbols_dict,
                                                            self.local_quandl_interest_rate_data_path],
                                         'ECONOMIC_INDICATORS_UNADJ': [
                                             self.quandl_ECONOMIC_INDICATORS_UNADJ_symbols_dict,
                                             self.local_quandl_economic_indicators_data_path],
                                         'ECONOMIC_INDICATORS_SEAS_ADJ': [
                                             self.quandl_ECONOMIC_INDICATORS_SEAS_ADJ_symbols_dict,
                                             self.local_quandl_economic_indicators_data_path],
                                         'FED_FORECASTS': [self.quandl_FED_FORECAST_symbols_dict,
                                                           self.local_quandl_fed_forecasts_data_path],
                                         'MISC': [self.local_misc_data_path],
                                         'STOCKS': [self.local_stock_data_path],
                                         'GDP': [self.gdp_sym_dict, self.path_to_local_gdp_data]}

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
        # write out the response to file
        with open(total_local_file_name, 'wb') as f:
            f.write(response.content)
        self.logger.info("QuandlSymbolInterface.download_quandl_stock_csv_file(): HTTP Response Status Code %s " + \
                         str(response.status_code))

    def daily_batch_equity_download(self):

        do = DataObject()
        suf_df = do.get_stock_universe_file_as_df()
        index_np_array = suf_df.index
        # do_dwnld = lambda idx : self.download_quandl_stock_csv_file(idx)
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
