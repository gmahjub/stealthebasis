import sys
from os.path import dirname, exists, join
#sys.path.append("C:\\Users\\ghazy\\workspace\\stealthebasis\\src\\")
from os_mux import OSMuxImpl
DATA_DIR = OSMuxImpl.get_proper_path("workspace/data/tiingo/stocks")

TICKERS = ['AAPL', 'GOOG', 'INTC', 'BRCM', 'YHOO']

def on_server_loaded(server_context):
    # pull the stock data from tiingo
    symbols = ['SPY', 'IWM', 'QQQ', 'FB', 'MSFT', 'AMZN', 'DIA', 'GLD', 'LQD', 'HYG', 'AAPL', 'ZROZ', 'IEF', 'TLT']
    
    if not all(exists(join(DATA_DIR, '%s.csv' % x)) for x in symbols):
        for x in symbols:
            if not exists(join(DATA_DIR, '%s.csv' % x)):
                print(join(DATA_DIR, '%s.csv' % x))
        print("Due to licensing considerations, you must first run download_sample_data.py to download this data set yourself.")
        print()

        sys.exit(1)

