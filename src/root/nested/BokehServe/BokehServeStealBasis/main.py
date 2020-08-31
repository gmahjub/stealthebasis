""" Create a simple stocks correlation dashboard.
Choose stocks to compare in the drop down widgets, and make selections
on the plots to update the summary and histograms accordingly.
.. note::
    Running this example requires downloading sample data. See
    the included `README`_ for more information.
Use the ``bokeh serve`` command to run the example by executing:
    bokeh serve stocks
at your command prompt. Then navigate to the URL
    http://localhost:5006/stocks
.. _README: https://github.com/bokeh/bokeh/blob/master/examples/app/stocks/README.md
"""
from functools import lru_cache
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, PreText, Select, HoverTool, Panel, LinearAxis, DataRange1d, CheckboxGroup, \
    TextInput
from bokeh.models.widgets import Tabs
from bokeh.plotting import figure
from bokeh.models import (Button, Slope, Div, ColumnDataSource, CustomJS, DataTable,
                          NumberFormatter, RangeSlider, TableColumn, )
from bokeh.palettes import Spectral5
from . import get_logger
from .os_mux import OSMuxImpl
from .process_data import ProcessData
from .data_interface import IEXTradingApi, DataProviderInterface
import pandas as pd
import numpy as np


class BokehHistogram:

    def __init__(self, colors=None, height=600, width=600):
        if colors is None:
            colors = ["SteelBlue", "Tan"]
        self.colors = colors
        self.height = height
        self.width = width

    def hist_hover(self, dataframe, col_name, bins=30, log_scale=False):
        hist, edges = np.histogram(dataframe[col_name], bins=bins)
        hist_df = pd.DataFrame({col_name: hist,
                                "left": edges[:-1],
                                "right": edges[1:]})
        hist_df["interval"] = ["%d to %d" % (left, right) for left, right in zip(hist_df["left"], hist_df["right"])]
        if log_scale:
            hist_df["log"] = np.log(hist_df[col_name])
            src = ColumnDataSource(hist_df)
            plot = figure(plot_height=self.height, plot_width=self.width,
                          title="Histogram of {}".format(col_name.capitalize()),
                          x_axis_label=col_name.capitalize(),
                          y_axis_label="Log Count")
            plot.quad(bottom=0, top="log", left="left",
                      right="right", source=src, fill_color=self.colors[0],
                      line_color="black", fill_alpha=0.7,
                      hover_fill_alpha=1.0, hover_fill_color=self.colors[1])
        else:
            src = ColumnDataSource(hist_df)
            plot = figure(plot_height=self.height, plot_width=self.width,
                          title="Histogram of {}".format(col_name.capitalize()),
                          x_axis_label=col_name.capitalize(),
                          y_axis_label="Count")
            plot.quad(bottom=0, top=col_name, left="left",
                      right="right", source=src, fill_color=self.colors[0],
                      line_color="black", fill_alpha=0.7,
                      hover_fill_alpha=1.0, hover_fill_color=self.colors[1])

        hover = HoverTool(tooltips=[('Interval', '@interval'),
                                    ('Count', str("@" + col_name))])
        plot.add_tools(hover)
        return plot

    def histotabs(self, dataframe, features, log_scale=False):
        hists = []
        for f in features:
            h = self.hist_hover(dataframe, f, log_scale=log_scale)
            p = Panel(child=h, title=f.capitalize())
            hists.append(p)
        t = Tabs(tabs=hists)
        return t

    def filtered_histotabs(self, dataframe, feature, filter_feature, log_scale=False):
        hists = []
        for col in dataframe[filter_feature].unique():
            sub_df = dataframe[dataframe[filter_feature] == col]
            histo = self.hist_hover(sub_df, feature, log_scale=log_scale)
            p = Panel(child=histo, title=col)
            hists.append(p)
        t = Tabs(tabs=hists)
        return t


DATA_DIR = OSMuxImpl.get_proper_path("/workspace/data/tiingo/stocks")
TRACK_STAT_MOM_DIR = OSMuxImpl.get_proper_path("/workspace/data/indicators/TrackStatMom/data/")
DEFAULT_TICKERS = []
procData = ProcessData()
LOGGER = get_logger()
BENCHMARK_TICKER_UNIVERSE_MAPPER = {'QQQ': 'NQ100',
                                    'SPY': 'SP500',
                                    'IWM': 'R2K',
                                    'DIA': 'DOW30'}
DATAPOINT_SIZES = list(range(6, 22, 3))
DATAPOINT_COLORS = Spectral5
N_DATAPOINT_SIZES = len(DATAPOINT_SIZES)
N_DATAPOINT_COLORS = len(DATAPOINT_COLORS)


def get_benchmarks():
    return ['SPY', 'QQQ', 'DIA', 'IWM']


def get_stock_universe(idx_name):
    LOGGER.info("begin running get_stock_universe(%s)...", str(idx_name))
    ticker_col_nm = 'Ticker'  # default for Russell files
    dpi = DataProviderInterface()
    ticker_mapped = BENCHMARK_TICKER_UNIVERSE_MAPPER[idx_name]
    pd_df = dpi.get_stock_universe_file_as_df(ticker_mapped)
    LOGGER.info("finished running get_stock_universe(%s)...", str(idx_name))
    return pd_df, ticker_col_nm


def get_index_components(idx_name):
    LOGGER.info("begin running get_index_components(%s)...", str(idx_name))
    stock_universe_df, ticker_col_nm = get_stock_universe(idx_name)
    symbols_list = list(stock_universe_df.index.dropna())
    symbols_list = symbols_list + get_benchmarks()
    LOGGER.info("finished running get_index_components(%s)...", str(idx_name))
    return symbols_list


def nix(val, lst):
    return [x for x in lst if x != val]


@lru_cache()
def load_target_ticker(ticker):
    LOGGER.info("begin running load_target_ticker(%s)...", str(ticker))
    stock_atr_df = procData.get_average_true_range(ticker=ticker, freq='W', window_size=12, start_date='2015-01-01')
    LOGGER.info("finished running load_target_ticker(%s)...", str(ticker))
    return stock_atr_df


@lru_cache()
def load_benchmark_ticker(ticker):
    LOGGER.info("begin running load_benchmark_ticker(%s)...", str(ticker))
    benchmark_atr_df = procData.get_average_true_range(ticker=ticker, freq='W', window_size=26, start_date='2015-01-01')
    LOGGER.info("finished running load_benchmark_ticker(%s)...", str(ticker))
    return benchmark_atr_df


@lru_cache()
def get_data(t1, t2):
    LOGGER.info("begin running get_data(%s, %s)...", str(t1), str(t2))
    stock_atr_df = load_target_ticker(t1)
    benchmark_atr_df = load_benchmark_ticker(t2)
    analyze_df = procData.get_atr_spread(stock_atr_df, benchmark_atr_df)
    analyze_df.dropna(inplace=True)
    LOGGER.info("finished running get_data(%s, %s)...", str(t1), str(t2))
    analyze_df.reset_index(inplace=True)
    return analyze_df


# set up widgets
# stats = PreText(text='', width=1500)
target_ticker = Select(title='Target Ticker', value='AAPL', options=get_index_components('SPY'))
benchmark_ticker = Select(title='Hedge (Benchmark) Ticker', value='SPY', options=get_benchmarks())

# set up the initial data
original_data = dict(date=[],
                     recentVol_emaAtr_diff_Atr=[],
                     ExcessFwdRets=[],
                     SpearmanCorr=[],
                     PearsonCorr=[],
                     r_squared=[],
                     intercept=[],
                     slope=[],
                     AbsValExcessFwdRets=[],
                     StockFwdRets=[],
                     BenchmarkFwdRets=[],
                     StockAdjClose=[],
                     BenchmarkAdjClose=[],
                     LineRegressTotal=[],
                     rSquaredTotal=[],
                     SlopeTotal=[],
                     InterceptTotal=[])
# set up plots
the_data = dict(date=[],
                recentVol_emaAtr_diff_Atr=[],
                ExcessFwdRets=[],
                SpearmanCorr=[],
                PearsonCorr=[],
                r_squared=[],
                intercept=[],
                slope=[],
                AbsValExcessFwdRets=[],
                StockFwdRets=[],
                BenchmarkFwdRets=[],
                StockAdjClose=[],
                BenchmarkAdjClose=[],
                LineRegressTotal=[],
                rSquaredTotal=[],
                SlopeTotal=[],
                InterceptTotal=[],
                FilteredLineRegressTotal=[],
                CumExcessFwdRets=[],
                CumStockFwdRets=[],
                CumBenchmarkFwdRets=[])
stats_data = dict(stat_field=[],
                  recentVol_emaAtr_diff_Atr=[],
                  ExcessFwdRets=[],
                  SpearmanCorr=[],
                  PearsonCorr=[],
                  intercept=[],
                  slope=[],
                  AbsValExcessFwdRets=[],
                  StockFwdRets=[],
                  BenchmarkFwdRets=[],
                  StockAdjClose=[],
                  BenchmarkAdjClose=[])
hist_data = dict(StockFwdRets=[],
                 BenchmarkFwdRets=[],
                 ExcessFwdRets=[],
                 StockFwdRetsLeft=[],
                 StockFwdRetsRight=[],
                 BenchmarkFwdRetsLeft=[],
                 BenchmarkFwdRetsRight=[],
                 ExcessFwdRetsLeft=[],
                 ExcessFwdRetsRight=[],
                 StockFwdRetsInterval=[],
                 BenchmarkFwdRetsInterval=[],
                 ExcessFwdRetsInterval=[])
modified_linreg_data = dict(original_total_gradient=[0.0],
                            original_total_yint=[0.0],
                            modified_total_gradient=[0.0],
                            modified_total_yint=[0.0])

original_source = ColumnDataSource(data=original_data)
source = ColumnDataSource(data=the_data)
source_static = ColumnDataSource(data=the_data)
stats_source = ColumnDataSource(data=stats_data)
linreg_data_source = ColumnDataSource(data=modified_linreg_data)
hist_source = ColumnDataSource(data=hist_data)
tools = 'pan,wheel_zoom,xbox_select,reset'

# scatter chart for linear regression (need to add still...)
corr = figure(plot_width=525, plot_height=525,
              tools='pan,wheel_zoom,box_select,reset')
corr.circle('StockFwdRets', 'BenchmarkFwdRets', size=3, source=source,
            selection_color="red", alpha=0.6, nonselection_alpha=0.1, selection_alpha=0.4)
orig_data_linreg = corr.line('StockFwdRets', 'LineRegressTotal', source=source, color='purple')
filtered_data_linreg = corr.line('StockFwdRets', 'FilteredLineRegressTotal', source=source, color='orange')
modified_slope_obj = Slope(gradient=linreg_data_source.to_df().iloc[0]['modified_total_gradient'],
                           y_intercept=linreg_data_source.to_df().iloc[0]['modified_total_yint'],
                           line_color='orange',
                           line_dash='dashed',
                           line_width=3.5)
# corr.add_layout(modified_slope_obj)

# first time series chart
ts1 = figure(plot_width=500, plot_height=400, tools=tools, x_axis_type='datetime', active_drag="xbox_select")
ts1.line('date', 'StockAdjClose', source=source_static)
ts1.circle('date', 'StockAdjClose', size=2, source=source, color=None, selection_color="red")

# second time series chart
ts2 = figure(plot_width=500, plot_height=400, tools=tools, x_axis_type='datetime', active_drag="xbox_select")
ts2.line('date', 'BenchmarkAdjClose', source=source_static)
ts2.circle('date', 'BenchmarkAdjClose', size=2, source=source, color=None, selection_color="red")

# set both x ranges to the same time range.
ts2.x_range = ts1.x_range

# third and fourth time series charts: vertical bar charts, split, mean return and std dev and sharpe of returns
ts3 = figure(plot_width=500, plot_height=400, tools=tools, x_axis_type='datetime', active_drag="xbox_select")
ts3.extra_y_ranges = {"foo_stk": DataRange1d(source.to_df()['CumStockFwdRets'].min(),
                                             source.to_df()['CumStockFwdRets'].max())}
ts3.add_layout(LinearAxis(y_range_name="foo_stk"), 'right')
# 3rd time series plot
ts3.vbar(x="date", top='StockFwdRets', color='blue', alpha=0.5, source=source, legend_label='StockFwdRets',
         y_range_name="foo_stk")
ts3.line('date', 'CumStockFwdRets', source=source, y_range_name='foo_stk')

# 4th time series plot
ts4 = figure(plot_width=500, plot_height=400, tools=tools, x_axis_type='datetime', active_drag="xbox_select")
ts4.extra_y_ranges = {"foo_bm": DataRange1d(source.to_df()['CumBenchmarkFwdRets'].min(),
                                            source.to_df()['CumBenchmarkFwdRets'].max())}
ts4.add_layout(LinearAxis(y_range_name="foo_bm"), 'right')
ts4.vbar(x='date', top='BenchmarkFwdRets', color="pink", source=source, legend_label='BenchmarkFwdRets',
         y_range_name="foo_bm")
ts4.line('date', 'CumBenchmarkFwdRets', source=source, y_range_name='foo_bm')

# 5th time series plot
ts5 = figure(plot_width=1000, plot_height=400, tools=tools, x_axis_type='datetime', active_drag="xbox_select")
ts5.extra_y_ranges = {"foo_exc": DataRange1d(source.to_df()['CumExcessFwdRets'].min(),
                                             source.to_df()['CumExcessFwdRets'].max())}
ts5.add_layout(LinearAxis(y_range_name="foo_exc"), 'right')
ts5.vbar(x='date', top='ExcessFwdRets', color='black', alpha=0.5, source=source, legend_label='ExcessFwdRets',
         y_range_name='foo_exc')
ts5.line('date', 'CumExcessFwdRets', source=source, y_range_name='foo_exc')

# Stock Histogram Plot
stock_hist_plot = figure(plot_height=500, plot_width=500, title="Histogram of Stock Fwd Returns",
                         x_axis_label='StockFwdRets', y_axis_label="Count")
stock_hist_plot.quad(bottom=0, top='StockFwdRets', left="StockFwdRetsLeft", right="StockFwdRetsRight",
                     source=hist_source, fill_color='SteelBlue', line_color="black", fill_alpha=0.7,
                     hover_fill_alpha=1.0, hover_fill_color='Tan')
stock_hist_hover = HoverTool(tooltips=[('Interval', '@StockFwdRetsInterval'), ('Count', '@StockFwdRets')])
stock_hist_plot.add_tools(stock_hist_hover)

# Benchmark Histogram Plot
benchmark_hist_plot = figure(plot_height=500, plot_width=500, title="Histogram of Benchmark Fwd Returns",
                             x_axis_label='BenchmarkFwdRets', y_axis_label="Count")
benchmark_hist_plot.quad(bottom=0, top='BenchmarkFwdRets', left="BenchmarkFwdRetsLeft", right="BenchmarkFwdRetsRight",
                         source=hist_source, fill_color='SteelBlue', line_color="black", fill_alpha=0.7,
                         hover_fill_alpha=1.0, hover_fill_color='Tan')
benchmark_hist_hover = HoverTool(tooltips=[('Interval', '@BenchmarkFwdRetsInterval'), ('Count', '@BenchmarkFwdRets')])
benchmark_hist_plot.add_tools(benchmark_hist_hover)

# Excess Returns Histogram Plot
excess_hist_plot = figure(plot_height=500, plot_width=500, title="Histogram of Excess Fwd Returns",
                          x_axis_label='ExcessFwdRets', y_axis_label="Count")
excess_hist_plot.quad(bottom=0, top='ExcessFwdRets', left="ExcessFwdRetsLeft", right="ExcessFwdRetsRight",
                      source=hist_source, fill_color='SteelBlue', line_color="black", fill_alpha=0.7,
                      hover_fill_alpha=1.0, hover_fill_color='Tan')
excess_hist_hover = HoverTool(tooltips=[('Interval', '@ExcessFwdRetsInterval'), ('Count', '@ExcessFwdRets')])
excess_hist_plot.add_tools(excess_hist_hover)


# Stock Returns Histogram plot - returned from the BokehHistogram.hist_hover() function is a figure object
# histogramStockReturns = bh.hist_hover(dataframe=source.to_df(), column='StockFwdRets', log_scale=False)
# Benchmark Returns Histogram plot
# histogramBechmarkReturns = bh.hist_hover(dataframe=source.to_df(), column='BenchmarkFwdRets', log_scale=False)
# Excess Returns Histogram plot
# histogramExcessReturns = bh.hist_hover(dataframe=source.to_df(), column='ExcessFwdRets', log_scale=False)

def create_figure(data_df=None):
    LOGGER.info("begin running create_figure()... ")
    if data_df is None:
        LOGGER.info("running create_figure() with dataframe set to NONE!")
        data_df = source.to_df()
        if len(data_df) == 0:
            LOGGER.error("retrieved dataframe from source but dataframe still empty!")
        else:
            LOGGER.info("retrieved dataframe from source, dataframe size = %s", str(len(data_df)))
    xs = data_df[independent_var_x.value].values
    ys = data_df[dependent_var_y.value].values
    x_title = independent_var_x.value.title()
    y_title = dependent_var_y.value.title()
    kw = dict()
    if independent_var_x.value in discrete_options:
        kw['independent_var_x_range'] = sorted(set(xs))
    if dependent_var_y in discrete_options:
        kw['dependent_var_y_range'] = sorted(set(ys))
    kw['title'] = "%s vs %s" % (x_title, y_title)
    p = figure(plot_height=750, plot_width=1000, tools='pan,box_zoom,hover,reset', **kw)
    p.xaxis.axis_label = x_title
    p.yaxis.axis_label = y_title

    if independent_var_x.value in discrete_options:
        p.xaxis.major_label_orientation = pd.np.pi / 4
    sz = 12
    if dataPointSize_filter.value != 'None':
        if len(set(data_df[dataPointSize_filter.value])) > N_DATAPOINT_SIZES:
            groups = pd.qcut(data_df[dataPointSize_filter.value].values, N_DATAPOINT_SIZES, duplicates='drop')
        else:
            groups = pd.Categorical(data_df[dataPointSize_filter.value])
        sz = [DATAPOINT_SIZES[xx] for xx in groups.codes]
    c = "#31AADE"
    if dataPointColor_filter.value != 'None':
        # data_df['ReturnCategory'] = data_df[dataPointColor_filter.value] > 0.0
        if len(set(data_df[dataPointColor_filter.value])) > N_DATAPOINT_COLORS:
            groups = pd.qcut(data_df[dataPointColor_filter.value].values, N_DATAPOINT_COLORS, duplicates='drop')
            groups = pd.qcut(data_df[dataPointColor_filter.value].values, 5, duplicates='drop')
        else:
            data_df['ReturnCategory'] = data_df[dataPointColor_filter.value] > 0.0
            groups = pd.Categorical(data_df[dataPointColor_filter.value])
        c = [DATAPOINT_COLORS[xx] for xx in groups.codes]
    p.circle(x=xs, y=ys, color=c, size=sz, line_color="black", alpha=0.6, hover_color='white', hover_alpha=0.5)
    LOGGER.info("finished running create_figure()...")
    return p


def update_hist_data(data_df, col_names=None, bins=30):
    if col_names is None:
        col_names = ['StockFwdRets', 'BenchmarkFwdRets', 'ExcessFwdRets']
    total_dict = dict()
    for col_name in col_names:
        hist, edges = np.histogram(data_df[col_name], bins=bins)
        hist_df = pd.DataFrame({col_name: hist, col_name + "Left": edges[:-1], col_name + "Right": edges[1:]})
        hist_df[col_name + "Interval"] = \
            ["%f to %f" % (left, right) for left, right in zip(hist_df[col_name + "Left"], hist_df[col_name + "Right"])]
        total_dict[col_name] = hist_df[col_name]
        total_dict[col_name + 'Left'] = hist_df[col_name + 'Left']
        total_dict[col_name + 'Right'] = hist_df[col_name + 'Right']
        total_dict[col_name + 'Interval'] = hist_df[col_name + 'Interval']
    hist_source.data = total_dict


def update(selected=None):
    LOGGER.info("begin running update()...")
    t1, t2 = target_ticker.value, benchmark_ticker.value
    LOGGER.info("in update(), tickers are target %s and hedge %s...", str(t1), str(t2))
    df_analyze = get_data(t1, t2)
    update_hist_data(df_analyze, bins=30)
    update_datatable(df_analyze)
    source.data = df_analyze
    original_source.data = df_analyze
    source_static.data = df_analyze
    update_stats(df_analyze, t1, t2)
    rSquaredTotalValue = df_analyze.rSquaredTotal.iloc[-1]
    corr.title.text = '%s returns vs. %s returns' % (t1, t2)
    corr.title.text += '\r\n R Squared (Entire Period) = %s' % str(round(rSquaredTotalValue, 3))
    corr.title.background_fill_color = "#aaaaee"
    ts1.title.text, ts2.title.text = t1, t2
    ts1.title.background_fill_color = "#aaaaee"
    ts2.title.background_fill_color = "#aaaaee"
    layout.children[2] = create_figure(df_analyze)
    reset_sliders_textbox()


def update_crossfilter(attrname, old, new):
    layout.children[2] = create_figure()


def update_stats(data, t1, t2):
    # stats.text = str(data[[t1, t2, t1 + '_returns', t2 + '_returns']].describe())
    LOGGER.info("running update_stats(%s, %s)...", str(t1), str(t2))
    update_stats_from_datatable(data)
    LOGGER.info("finished update_stats(%s, %s)...", str(t1), str(t2))


# set up callbacks
def target_ticker_change(attrname, old, new):
    # ticker2.options = nix(new, DEFAULT_TICKERS)
    LOGGER.info("invoked callback function target_ticker_change()...")
    update()


def benchmark_ticker_change(attrname, old, new):
    target_ticker.options = get_index_components(new)
    # ticker1.options = nix(new, DEFAULT_TICKERS)
    LOGGER.info("invoked callback function benchmark_ticker_change()...")
    update()


target_ticker.on_change('value', target_ticker_change)
benchmark_ticker.on_change('value', benchmark_ticker_change)


def selection_change(attrname, old, new):
    LOGGER.info("invoked selection_change() callback function...")
    t1, t2 = target_ticker.value, benchmark_ticker.value
    LOGGER.info("invoked selection_change(): target ticker: %s, hedge ticker: %s", t1, t2)
    data = get_data(t1, t2)
    selected_data = data
    selected = source.selected.indices
    if selected:
        selected_data = data.iloc[selected, :]
    update_stats(selected_data, t1, t2)
    update_corr_chart(selected_data, data)
    update_hist_data(selected_data)


def update_corr_chart(selected_data_df, data_df):
    the_selectedData_lrModel = procData.get_linear_regression(selected_data_df, 'StockFwdRets', 'BenchmarkFwdRets')
    gradient = the_selectedData_lrModel['np.poly.slope']
    y_int = the_selectedData_lrModel['np.poly.intercept']
    data_df['FilteredLineRegressGradient'] = data_df.SlopeTotal
    data_df['FilteredLineRegressYint'] = data_df.InterceptTotal
    data_df['FilteredLineRegressTotal'] = data_df.StockFwdRets.mul(gradient).add(y_int)
    source.data = {
        'date': data_df.date,
        'recentVol_emaAtr_diff_Atr': data_df.recentVol_emaAtr_diff_Atr,
        'ExcessFwdRets': data_df.ExcessFwdRets,
        'SpearmanCorr': data_df.SpearmanCorr,
        'PearsonCorr': data_df.PearsonCorr,
        'intercept': data_df.intercept,
        'slope': data_df.slope,
        'AbsValExcessFwdRets': data_df.AbsValExcessFwdRets,
        'StockFwdRets': data_df.StockFwdRets,
        'BenchmarkFwdRets': data_df.BenchmarkFwdRets,
        'StockAdjClose': data_df.StockAdjClose,
        'BenchmarkAdjClose': data_df.BenchmarkAdjClose,
        'LineRegressTotal': data_df.LineRegressTotal,
        'rSquaredTotal': data_df.rSquaredTotal,
        'SlopeTotal': data_df.SlopeTotal,
        'InterceptTotal': data_df.InterceptTotal,
        'StockLogRets': data_df.StockLogRets,
        'BenchmarkLogRets': data_df.BenchmarkLogRets,
        'FilteredLineRegressGradient': data_df.FilteredLineRegressGradient,
        'FilteredLineRegressYint': data_df.FilteredLineRegressYint,
        'FilteredLineRegressTotal': data_df.FilteredLineRegressTotal,
        'CumExcessFwdRets': data_df.CumExcessFwdRets,
        'CumStockFwdRets': data_df.CumStockFwdRets,
        'CumBenchmarkFwdRets': data_df.CumBenchmarkFwdRets
    }


def reset_sliders_textbox():
    atr_slider.value = (-10.0, 10.0)
    low_atr_range.value = "-10.0"
    high_atr_range.value = "10.0"
    spear_rank_slider.value = (-1.0, 1.0)


def update_datatable_from_textbox(data_df):
    atr_slider.value = (float(low_atr_range.value), float(high_atr_range.value))
    update_datatable(data_df=data_df)


def update_datatable(data_df):
    """
        Index(['recentVol_emaAtr_diff_Atr', 'ExcessFwdRets', 'SpearmanCorr',
               'PearsonCorr', 'r_squared', 'intercept', 'slope', 'AbsValExcessFwdRets',
               'StockFwdRets', 'BenchmarkFwdRets', 'StockAdjClose',
               'BenchmarkAdjClose'],
              dtype='object')
    """
    print("low atr range value", low_atr_range.value)
    print("high atr range value", high_atr_range.value)

    LOGGER.info("running update_datatable()...")
    if data_df is None:
        data_df = original_source.to_df()
    df_analyze = data_df
    df_analyze['FilteredLineRegressGradient'] = df_analyze.SlopeTotal
    df_analyze['FilteredLineRegressYint'] = df_analyze.InterceptTotal
    df_analyze['FilteredLineRegressTotal'] = df_analyze.LineRegressTotal
    current = df_analyze[(df_analyze.recentVol_emaAtr_diff_Atr >= atr_slider.value[0]) &
                         (df_analyze.recentVol_emaAtr_diff_Atr <= atr_slider.value[1])]
    current.reset_index(inplace=True)
    the_lr_model = procData.get_linear_regression(current, 'StockFwdRets', 'BenchmarkFwdRets')
    gradient = the_lr_model['np.poly.slope']
    y_int = the_lr_model['np.poly.intercept']
    df_analyze.loc[(df_analyze.recentVol_emaAtr_diff_Atr >= atr_slider.value[0]) &
                   (df_analyze.recentVol_emaAtr_diff_Atr <= atr_slider.value[
                       1]), ['FilteredLineRegressGradient']] = gradient
    df_analyze.loc[(df_analyze.recentVol_emaAtr_diff_Atr >= atr_slider.value[0]) &
                   (df_analyze.recentVol_emaAtr_diff_Atr <= atr_slider.value[
                       1]), ['FilteredLineRegressYint']] = y_int
    df_analyze.loc[(df_analyze.recentVol_emaAtr_diff_Atr >= atr_slider.value[0]) &
                   (df_analyze.recentVol_emaAtr_diff_Atr <= atr_slider.value[
                       1]), ['FilteredLineRegressTotal']] = df_analyze.StockFwdRets.mul(gradient).add(y_int)
    current = df_analyze[(df_analyze.recentVol_emaAtr_diff_Atr >= atr_slider.value[0]) &
                         (df_analyze.recentVol_emaAtr_diff_Atr <= atr_slider.value[1]) &
                         (df_analyze.SpearmanCorr >= spear_rank_slider.value[0]) &
                         (df_analyze.SpearmanCorr <= spear_rank_slider.value[1])]
    current.CumStockFwdRets = (1 + current.StockFwdRets).cumprod() - 1
    current.CumBenchmarkFwdRets = (1 + current.BenchmarkFwdRets).cumprod() - 1
    current.CumExcessFwdRets = (1 + current.ExcessFwdRets).cumprod() - 1
    current.reset_index(inplace=True)
    LOGGER.info("in update_datatable(): new slope with filtered points: %s", gradient)
    LOGGER.info("in update_datatable(): new y_int with filtered points: %s", y_int)

    source.data = {
        'date': current.date,
        'recentVol_emaAtr_diff_Atr': current.recentVol_emaAtr_diff_Atr,
        'ExcessFwdRets': current.ExcessFwdRets,
        'SpearmanCorr': current.SpearmanCorr,
        'PearsonCorr': current.PearsonCorr,
        'intercept': current.intercept,
        'slope': current.slope,
        'AbsValExcessFwdRets': current.AbsValExcessFwdRets,
        'StockFwdRets': current.StockFwdRets,
        'BenchmarkFwdRets': current.BenchmarkFwdRets,
        'StockAdjClose': current.StockAdjClose,
        'BenchmarkAdjClose': current.BenchmarkAdjClose,
        'LineRegressTotal': current.LineRegressTotal,
        'rSquaredTotal': current.rSquaredTotal,
        'SlopeTotal': current.SlopeTotal,
        'InterceptTotal': current.InterceptTotal,
        'StockLogRets': current.StockLogRets,
        'BenchmarkLogRets': current.BenchmarkLogRets,
        'FilteredLineRegressGradient': current.FilteredLineRegressGradient,
        'FilteredLineRegressYint': current.FilteredLineRegressYint,
        'FilteredLineRegressTotal': current.FilteredLineRegressTotal,
        'CumStockFwdRets': current.CumStockFwdRets,
        'CumBenchmarkFwdRets': current.CumBenchmarkFwdRets,
        'CumExcessFwdRets': current.CumExcessFwdRets
    }
    linreg_data_source.data = {
        'original_total_gradient': current.SlopeTotal,
        'original_total_yint': current.InterceptTotal,
        'modified_total_gradient': current.FilteredLineRegressGradient,
        'modified_total_yint': current.FilteredLineRegressYint
    }
    """
    source_static.data = {
        'date': current.date,
        'recentVol_emaAtr_diff_Atr': current.recentVol_emaAtr_diff_Atr,
        'ExcessFwdRets': current.ExcessFwdRets,
        'SpearmanCorr': current.SpearmanCorr,
        'PearsonCorr': current.PearsonCorr,
        'intercept': current.intercept,
        'slope': current.slope,
        'AbsValExcessFwdRets': current.AbsValExcessFwdRets,
        'StockFwdRets': current.StockFwdRets,
        'BenchmarkFwdRets': current.BenchmarkFwdRets,
        'StockAdjClose': current.StockAdjClose,
        'BenchmarkAdjClose': current.BenchmarkAdjClose
    }
    """
    update_stats_from_datatable(current)
    update_hist_data(current)
    LOGGER.info("finished running update_datatable()....")


def update_stats_from_datatable(data):
    """
    Update statistics from the datatable updated source.
    :param data:
    :return:
    """
    LOGGER.info("running update_stats_from_datatable()...")
    data_described = data.describe()
    stats_source.data = {
        'stat_field': data_described.index,
        'recentVol_emaAtr_diff_Atr': data_described.recentVol_emaAtr_diff_Atr,
        'ExcessFwdRets': data_described.ExcessFwdRets,
        'SpearmanCorr': data_described.SpearmanCorr,
        'PearsonCorr': data_described.PearsonCorr,
        'intercept': data_described.intercept,
        'slope': data_described.slope,
        'AbsValExcessFwdRets': data_described.AbsValExcessFwdRets,
        'StockFwdRets': data_described.StockFwdRets,
        'BenchmarkFwdRets': data_described.BenchmarkFwdRets,
        'StockAdjClose': data_described.StockAdjClose,
        'BenchmarkAdjClose': data_described.BenchmarkAdjClose
    }
    LOGGER.info("finished running update_stats_from_datatable()...")


def cross_filter_button_handler():
    layout.children[2] = create_figure()


# set up filtering/grouping widgets.
var_x_y_options = sorted(source.to_df().columns)
discrete_options = [x for x in var_x_y_options if source.to_df()[x].dtype == object]
continuous_options = [x for x in var_x_y_options if x not in discrete_options]
independent_var_x = Select(title='Independent Variable', value='SpearmanCorr', options=sorted(source.to_df().columns))
independent_var_x.on_change('value', update_crossfilter)
dependent_var_y = Select(title='Dependent Variable', value='recentVol_emaAtr_diff_Atr',
                         options=sorted(source.to_df().columns))
dependent_var_y.on_change('value', update_crossfilter)
dataPointSize_filter = Select(title='Size', value='None', options=['None'] + continuous_options)
dataPointSize_filter.on_change('value', update_crossfilter)
dataPointColor_filter = Select(title='Color', value='None', options=['None'] + continuous_options)
dataPointColor_filter.on_change('value', update_crossfilter)
atr_slider = RangeSlider(title='Max Volatilty (ATR) Spread', start=-10.0, end=10.0, value=(-10.0, 10.0), step=0.1)
low_atr_range = TextInput(value="-10.0", title="Low ATR Range")
high_atr_range = TextInput(value="10.0", title="High ATR Range")
spear_rank_slider = RangeSlider(title='Spearman Rank Correlation, Select Range', start=-1.0, end=1.0, value=(-1.0, 1.0),
                                step=0.01)
atr_slider.on_change("value", lambda attr, old, new: update_datatable(data_df=None))
spear_rank_slider.on_change("value", lambda attr, old, new: update_datatable(data_df=None))
crossFilter_refreshButton = Button(label="Apply Cross-Filter Analyzer", button_type="success")
crossFilter_refreshButton.on_click(cross_filter_button_handler)
resetButton = Button(label="Reset All", button_type="danger")
resetButton.on_click(update)
columns = [
    TableColumn(field="date", title="Date"),
    TableColumn(field="recentVol_emaAtr_diff_Atr", title="Volatility (ATR) Spread"),
    TableColumn(field="ExcessFwdRets", title="Excess Forward Return"),
    TableColumn(field="SpearmanCorr", title="Spearman Rank Correlation"),
    TableColumn(field="PearsonCorr", title="Pearson Correlation"),
    TableColumn(field="intercept", title="Line Regress Intercept"),
    TableColumn(field="slope", title="Line Regress Slope"),
    TableColumn(field="AbsValExcessFwdRets", title="Magnitude of Returns"),
    TableColumn(field="StockFwdRets", title='Target Forward Return'),
    TableColumn(field="BenchmarkFwdRets", title='Benchmark Forward Return'),
    TableColumn(field="StockAdjClose", title='Target Px. (Adj Close)'),
    TableColumn(field="BenchmarkAdjClose", title='Benchmark Px. (Adj Close)')
]

stats_columns = [
    TableColumn(field="stat_field", title='Aggregation Type'),
    TableColumn(field="recentVol_emaAtr_diff_Atr", title="Volatility (ATR) Spread"),
    TableColumn(field="ExcessFwdRets", title="Excess Fwd Rets"),
    TableColumn(field="SpearmanCorr", title="Spearman Rank Correlation"),
    TableColumn(field="PearsonCorr", title="Pearson Correlation"),
    TableColumn(field="intercept", title="Line Regress Intercept"),
    TableColumn(field="slope", title="Line Regress Slope"),
    TableColumn(field="AbsValExcessFwdRets", title="Magnitude Rets"),
    TableColumn(field="StockFwdRets", title="Target Fwd Rets"),
    TableColumn(field="BenchmarkFwdRets", title="Benchmark Fwd Rets"),
    TableColumn(field="StockAdjClose", title="Target Px. (Adj Close)"),
    TableColumn(field="BenchmarkAdjClose", title="Benchmark Px (Adj Close)")
]
agg_stats_title_div = Div(text="""<b>Aggregated Statistics, Selected Data</b> """, width=1000, height=20,
                          background="yellow")
data_table = DataTable(source=source, columns=columns, width=1000)
ts_data_corrAnalysis_title_div = Div(text="""<b>Correlation Analysis - Time Series Data</b> """, width=1000, height=20,
                                     background="yellow")
stats_data_table = DataTable(source=stats_source, columns=stats_columns, height=250, width=1000)
period_cum_returns_title_div = Div(text="""<b>Periodic & Cummulative Returns, Selected Data</b> """, width=1000,
                                   height=20, background="yellow")
histograms_title_div = Div(text="""<b>Histograms, Selected Data</b> """, width=1000, height=20, background="yellow")
crossFilterAnalysis_title_div = Div(text="""<b>Cross-Filter Analyzer</b> """, width=1000, height=20,
                                    background="yellow")
flat_price_charts_title_div = Div(text="""<b>Flat Price Charts</b>""", width=1000, height=20, background="yellow")
source.selected.on_change('indices', selection_change)

controls = [low_atr_range, high_atr_range]
for control in controls:
    control.on_change('value', lambda attr, old, new: update_datatable_from_textbox(data_df=None))

slider_dataTable = row(data_table)
stats_dataTable = row(stats_data_table)
# set up layout
widgets = column(target_ticker, benchmark_ticker, atr_slider, spear_rank_slider, independent_var_x, dependent_var_y,
                 dataPointSize_filter, dataPointColor_filter, crossFilter_refreshButton)
atr_text_inputters = column(low_atr_range, high_atr_range, resetButton)
main_row = row(corr, widgets, atr_text_inputters)
price_charts_row = row(ts1, ts2)
rets_vbar_charts_row = row(ts3, ts4)
excess_rets_vbar_charts_row = row(ts5)
outright_hist_plot_row = row(stock_hist_plot, benchmark_hist_plot)
excess_hist_plot_row = row(excess_hist_plot)
series = column(agg_stats_title_div, stats_dataTable, flat_price_charts_title_div, price_charts_row)
layout = column(main_row, crossFilterAnalysis_title_div, create_figure(), series, ts_data_corrAnalysis_title_div,
                data_table, period_cum_returns_title_div,
                rets_vbar_charts_row, excess_rets_vbar_charts_row, histograms_title_div,
                outright_hist_plot_row, excess_hist_plot_row)

# initialize
update()

curdoc().add_root(layout)
curdoc().title = "Correlation Evaluator"
