from bokeh.document import Document
from bokeh.models import (ColumnDataSource, DataRange1d, Plot, LinearAxis, Grid, Circle, HoverTool, BoxSelectTool,
                          Title, Button, CustomJS, RangeSlider, TableColumn,)
from bokeh.models.widgets import DataTable, TableColumn, StringFormatter, NumberFormatter, StringEditor, IntEditor, \
    NumberEditor, SelectEditor, DateFormatter
from bokeh.models.layouts import Column
from bokeh.embed import file_html
from bokeh.resources import INLINE
from bokeh.util.browser import view
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.sampledata.autompg2 import autompg2 as mpg
from root.nested.SysOs.os_mux import OSMuxImpl
from root.nested import get_logger

LOGGER = get_logger()


class ExtendDataTable:

    @staticmethod
    def make_analyze_datatable_bokeh_server(df_analyze):
        """
        df_analyze = pd.concat([new_df.recentVol_emaAtr_diff_Atr, new_df.ExcessFwdRets, df_currCurr.SpearmanCorr,
                                df_currCurr.PearsonCorr, df_currCurr.r_squared, df_currCurr.intercept,
                                df_currCurr.slope, new_df.AbsValExcessFwdRets, new_df.StockFwdRets,
                                new_df.BenchmarkFwdRets, new_df.StockAdjClose, new_df.BenchmarkAdjClose], axis=1)
        Index([ 'recentVol_emaAtr_diff_Atr', 'ExcessFwdRets', 'SpearmanCorr',
                'PearsonCorr', 'r_squared', 'intercept', 'slope', 'AbsValExcessFwdRets',
                'StockFwdRets', 'BenchmarkFwdRets', 'StockAdjClose',
                'BenchmarkAdjClose'],
        dtype='object')
        """
        source = ColumnDataSource(df_analyze)
        columns = [
            TableColumn(field="recentVol_emaAtr_diff_Atr", title="Volatility (ATR) Spread"),
            TableColumn(field="ExcessFwdRets", title="Excess Forward Return"),
            TableColumn(field="SpearmanCorr", title="Spearman Rank Correlation"),
            TableColumn(field="PearsonCorr", title="Pearson Correlation"),
            TableColumn(field="r_squared", title="R-Squared"),
            TableColumn(field="intercept", title="Line Regress Intercept"),
            TableColumn(field="slope", title="Line Regress Slope"),
            TableColumn(field="AbsValExcessFwdRets", title="Magnitude of Returns"),
            TableColumn(field="StockFwdRets", title='Target Forward Return'),
            TableColumn(field="BenchmarkFwdRets", title='Benchmark Forward Return'),
            TableColumn(field="StockAdjClose", title='Target Px. (Adj Close)'),
            TableColumn(field="BenchmarkAdjClose", title='Benchmark Px. (Adj Close)')
        ]
        data_table = DataTable(source=source, columns=columns, widht=1000)
        return data_table

    @staticmethod
    def make_timeseries_datatable_bokeh_server(df):
        source = ColumnDataSource(df)
        columns = [
            TableColumn(field="Date", title="Date", formatter=DateFormatter()),
            TableColumn(field="Target", title="Target Timeseries",
                        formatter=StringFormatter(font_style="bold", text_color='red')),
            TableColumn(field="Hedge", title="Hedge Timeseries",
                        formatter=StringFormatter(font_style="bold", text_color='blue')),
            TableColumn(field="Correlation", title="Correlation",
                        formatter=StringFormatter(font_style="bold", text_color='darkgreen'))
        ]
        data_table = DataTable(source=source, columns=columns, width=1000)
        return data_table

    @staticmethod
    def make_timeseries_datatable_plot_bokeh_server(df):
        source = ColumnDataSource(df)
        plot = Plot(title=Title(text="Rolling Spearman Rank Correlation)", align="center"),
                    x_range=DataRange1d(),
                    y_range=DataRange1d(), plot_width=1000, plot_height=300)
        plot.add_layout(LinearAxis(), 'below')
        yaxis = LinearAxis()
        plot.add_layout(yaxis, 'left')
        plot.add_layout(Grid(dimension=1, ticker=yaxis.ticker))
        # Add Glyphs
        correlation_glyph = Circle(x="Date", y="Correlation", fill_color="#396285", size=8, fill_alpha=0.5,
                                   line_alpha=0.5)
        target_glyph = Circle(x="Date", y="Target", fill_color="#396285", size=8, fill_alpha=0.5,
                              line_alpha=0.5)
        hedge_glyph = Circle(x="Date", y="Hedge", fill_color="#396285", size=8, fill_alpha=0.5,
                             line_alpha=0.5)
        correlation = plot.add_glyph(source, correlation_glyph)
        target = plot.add_glyph(source, target_glyph)
        hedge = plot.add_glyph(source, hedge_glyph)
        # Add the tools
        tooltips = [
            ("Date", "@Date"),
            ("Correlation", "@Correlation"),
            ("Target", "@Target"),
            ("Hedge", "@Hedge")
        ]
        correlation_hover_tool = HoverTool(renderers=[correlation], tooltips=tooltips)
        target_hover_tool = HoverTool(renderers=[target], tooltips=tooltips)
        hedge_hover_tool = HoverTool(renderers=[hedge], tooltips=tooltips)
        select_tool = BoxSelectTool(renderers=[target, hedge, correlation], dimensions='width')
        plot.add_tools(target_hover_tool, hedge_hover_tool, correlation_hover_tool, select_tool)
        return plot

    @staticmethod
    def make_correlation_datatable(correl_df):
        """
        the input datframe must have columns ['Target (as string)', 'Hedge (as string)', 'Correlation ( as float )']
        :param correl_df:
        :return:
        """
        correl_df.reset_index(inplace=True)
        source = ColumnDataSource(correl_df)
        target_ts_asset = sorted(correl_df["Target"].unique())
        hedge_ts_asset = sorted(correl_df["Hedge"].unique())
        columns = [
            TableColumn(field="Target", title="Target Timeseries",
                        formatter=StringFormatter(font_style="bold", text_color='red')),
            TableColumn(field="Hedge", title="Hedge Timeseries",
                        formatter=StringFormatter(font_style="bold", text_color='blue')),
            TableColumn(field="Correlation", title="Correlation",
                        formatter=StringFormatter(font_style="bold", text_color='darkgreen'))
        ]
        data_table = DataTable(source=source, columns=columns, editable=False, width=1000)
        plot = Plot(title=Title(text="Correlations, Target vs. Hedge Timeseries)", align="center"),
                    x_range=DataRange1d(),
                    y_range=DataRange1d(), plot_width=1000, plot_height=300)
        # Set up x & y axis
        plot.add_layout(LinearAxis(), 'below')
        yaxis = LinearAxis()
        plot.add_layout(yaxis, 'left')
        plot.add_layout(Grid(dimension=1, ticker=yaxis.ticker))
        # Add Glyphs
        correlation_glyph = Circle(x="index", y="Correlation", fill_color="#396285", size=8, fill_alpha=0.5,
                                   line_alpha=0.5)
        target_glyph = Circle(x="index", y="Target", fill_color="#396285", size=8, fill_alpha=0.5,
                              line_alpha=0.5)
        hedge_glyph = Circle(x="index", y="Hedge", fill_color="#396285", size=8, fill_alpha=0.5,
                             line_alpha=0.5)
        correlation = plot.add_glyph(source, correlation_glyph)
        target = plot.add_glyph(source, target_glyph)
        hedge = plot.add_glyph(source, hedge_glyph)
        # Add the tools
        tooltips = [
            ("Correlation", "@Correlation"),
            ("Target", "@Target"),
            ("Hedge", "@Hedge")
        ]
        correlation_hover_tool = HoverTool(renderers=[correlation], tooltips=tooltips)
        target_hover_tool = HoverTool(renderers=[target], tooltips=tooltips)
        hedge_hover_tool = HoverTool(renderers=[hedge], tooltips=tooltips)
        select_tool = BoxSelectTool(renderers=[target, hedge, correlation], dimensions='width')
        plot.add_tools(target_hover_tool, hedge_hover_tool, correlation_hover_tool, select_tool)
        layout = Column(plot, data_table)
        the_doc = Document()
        the_doc.add_root(layout)
        return the_doc

    @staticmethod
    def make_example_datatable():
        source = ColumnDataSource(mpg)
        print(source.column_names)
        manufacturers = sorted(mpg["manufacturer"].unique())
        models = sorted(mpg["model"].unique())
        transmissions = sorted(mpg["trans"].unique())
        drives = sorted(mpg["drv"].unique())
        classes = sorted(mpg["class"].unique())

        columns = [
            TableColumn(field="manufacturer", title="Manufacturer", editor=SelectEditor(options=manufacturers),
                        formatter=StringFormatter(font_style="bold")),
            TableColumn(field="model", title="Model", editor=StringEditor(completions=models)),
            TableColumn(field="displ", title="Displacement", editor=NumberEditor(step=0.1),
                        formatter=NumberFormatter(format="0.0")),
            TableColumn(field="year", title="Year", editor=IntEditor()),
            TableColumn(field="cyl", title="Cylinders", editor=IntEditor()),
            TableColumn(field="trans", title="Transmission", editor=SelectEditor(options=transmissions)),
            TableColumn(field="drv", title="Drive", editor=SelectEditor(options=drives)),
            TableColumn(field="class", title="Class", editor=SelectEditor(options=classes)),
            TableColumn(field="cty", title="City MPG", editor=IntEditor()),
            TableColumn(field="hwy", title="Highway MPG", editor=IntEditor()),
        ]
        data_table = DataTable(source=source, columns=columns, editable=True, width=1000)
        plot = Plot(title=None, x_range=DataRange1d(), y_range=DataRange1d(), plot_width=1000, plot_height=300)
        # Set up x & y axis
        plot.add_layout(LinearAxis(), 'below')
        yaxis = LinearAxis()
        plot.add_layout(yaxis, 'left')
        plot.add_layout(Grid(dimension=1, ticker=yaxis.ticker))

        # Add Glyphs
        cty_glyph = Circle(x="index", y="cty", fill_color="#396285", size=8, fill_alpha=0.5, line_alpha=0.5)
        hwy_glyph = Circle(x="index", y="hwy", fill_color="#CE603D", size=8, fill_alpha=0.5, line_alpha=0.5)
        cty = plot.add_glyph(source, cty_glyph)
        hwy = plot.add_glyph(source, hwy_glyph)

        # Add the tools
        tooltips = [
            ("Manufacturer", "@manufacturer"),
            ("Model", "@model"),
            ("Displacement", "@displ"),
            ("Year", "@year"),
            ("Cylinders", "@cyl"),
            ("Transmission", "@trans"),
            ("Drive", "@drv"),
            ("Class", "@class"),
        ]
        cty_hover_tool = HoverTool(renderers=[cty], tooltips=tooltips + [("City MPG", "@cty")])
        hwy_hover_tool = HoverTool(renderers=[hwy], tooltips=tooltips + [("Highway MPG", "@hwy")])
        select_tool = BoxSelectTool(renderers=[cty, hwy], dimensions='width')
        plot.add_tools(cty_hover_tool, hwy_hover_tool, select_tool)
        layout = Column(plot, data_table)
        doc = Document()
        doc.add_root(layout)
        return doc

    @staticmethod
    def validate_show_document(html_document, html_filename, html_dir, viewHtml=False):
        html_document.validate()
        proper_dir = OSMuxImpl.get_proper_path(html_dir)
        proper_filename = proper_dir + html_filename
        with open(proper_filename, "w", encoding='utf-8') as f:
            f.write(file_html(html_document, INLINE, "Data Tables"))
        LOGGER.info("extend_bokeh_datatables.ExtendBokeh.validate_show_document(): wrote %s in dir %s ",
                    html_filename, proper_dir)
        if viewHtml is not False:
            view(proper_filename)


if __name__ == "__main__":
    doc = ExtendDataTable.make_example_datatable()
    doc.validate()
    dir_name = "workspace/data/bokeh/html/"
    filename = OSMuxImpl.get_proper_path(dir_name) + "data_tables.html"
    with open(filename, "w", encoding='utf-8') as f:
        f.write(file_html(doc, INLINE, "Data Tables"))
    print("Wrote %s" % filename)
    view(filename)
