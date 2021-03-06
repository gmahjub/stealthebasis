from bokeh.plotting import figure, ColumnDataSource
from bokeh.io import output_file, show, save
from bokeh.layouts import row, column, gridplot, widgetbox
from bokeh.models.widgets import Panel, Tabs, DataTable, DateFormatter, TableColumn, StringFormatter, \
    HTMLTemplateFormatter, NumberFormatter, StringEditor, IntEditor, NumberEditor, SelectEditor
from bokeh.models import HoverTool, Title, BoxAnnotation, Span, Range1d, DataRange1d, Plot, LinearAxis, \
    Grid, Circle, Line, HoverTool, BoxSelectTool
from bokeh.models import CategoricalColorMapper, Band, ColumnDataSource
from bokeh.models.layouts import Column
from bokeh.models.axes import LinearAxis
from bokeh.core.properties import value
from bokeh.document import Document
from bokeh.embed import file_html
from bokeh.resources import INLINE
from bokeh.util.browser import view
from bokeh.palettes import Spectral5
from bokeh.transform import factor_cmap

from collections import OrderedDict
import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats

from root.nested.statisticalAnalysis.hacker_stats import HackerStats
from root.nested.statisticalAnalysis.ecdf import ECDF
from root.nested import get_logger


class ExtendBokeh(object):
    """description of class"""

    LOGGER = get_logger()

    def __init__(self, **kwargs):

        # self.logger = get_logger()
        return super().__init__(**kwargs)

    @staticmethod
    def visualizeCbo(df,
                     title,
                     xlabel,
                     ylabel,
                     html_output_filename):

        fieldnames = [col_nm.replace('\n', '').lower().replace(' ', '') for col_nm in df.columns]
        df.columns = fieldnames
        source = ColumnDataSource(df)
        print(df.columns)
        ind_income_taxes = sorted(df["individualincometaxes"].unique())
        payroll_taxes = sorted(df["payrolltaxes"].unique())
        corporate_income_taxes = sorted(df["corporateincometaxes"].unique())
        excise_taxes = sorted(df["excisetaxes"].unique())
        estate_gift_taxes = sorted(df["estateandgifttaxes"].unique())
        customs_duties = sorted(df["customsduties"].unique())
        misc_receipts = sorted(df["miscellaneousreceipts"].unique())
        total = sorted(df["total"].unique())
        # year = sorted(df["Year"].unique())
        first_year = pd.to_numeric(df.year).min()
        last_year = pd.to_numeric(df.year).max()

        columns = [
            TableColumn(field="year", title="Year",
                        editor=NumberEditor()),
            TableColumn(field="individualincometaxes", title="Individual Income Taxes",
                        editor=NumberEditor()),
            TableColumn(field="payrolltaxes", title="Payroll Taxes",
                        editor=NumberEditor(),
                        formatter=NumberFormatter(format="0.0")),
            TableColumn(field="corporateincometaxes", title="Corporate Income Taxes",
                        editor=NumberEditor(),
                        formatter=NumberFormatter(format="0.0")),
            TableColumn(field="excisetaxes", title="Excise Taxes",
                        editor=NumberEditor(),
                        formatter=NumberFormatter(format="0.0")),
            TableColumn(field="estateandgifttaxes", title="Estate and Gift Taxes",
                        editor=NumberEditor()),
            TableColumn(field="customsduties", title="Customs Duties",
                        editor=NumberEditor()),
            TableColumn(field="miscellaneousreceipts", title="Miscellaneous Receipts",
                        editor=NumberEditor()),
            TableColumn(field="total", title="Total",
                        editor=NumberEditor()),
        ]
        data_table = DataTable(source=source, columns=columns, editable=False, width=1300)
        year_range = Range1d(int(first_year) - 1, int(last_year) + 1)
        title_obj = Title()
        title_obj.text = "Congressional Budget Office, CBO.gov: " + title
        plot = Plot(title=title_obj, x_range=year_range, y_range=DataRange1d(), plot_width=1300, plot_height=600)
        # Set up x & y axis
        plot.add_layout(LinearAxis(), 'below')
        yaxis = LinearAxis()
        plot.add_layout(yaxis, 'left')
        plot.add_layout(Grid(dimension=1, ticker=yaxis.ticker))
        plot.xaxis.axis_label = xlabel  # "Year"
        plot.yaxis.axis_label = ylabel  # "In Billions (US $)"

        # Add Glyphs
        ind_inc_tax_glyph = Line(x="index", y="individualincometaxes", line_width=5, line_color='orange')
        payroll_taxes_glyph = Line(x="index", y="payrolltaxes", line_width=4, line_color='blue')
        corporate_income_taxes_glyph = Line(x="index", y="corporateincometaxes", line_width=3, line_color='green')
        excise_taxes_glyph = Line(x="index", y="excisetaxes", line_width=2, line_color='yellow')
        estate_and_gift_taxes_glyph = Line(x="index", y="estateandgifttaxes", line_width=1, line_color='magenta')
        customs_duties_glyph = Line(x="index", y="customsduties", line_width=1, line_color='limegreen')
        misc_receipts_glyph = Line(x="index", y="miscellaneousreceipts", line_width=1, line_color='pink')
        total_glyph = Line(x="index", y="total", line_color='red', line_width=6)
        ind_inc_tax = plot.add_glyph(source, ind_inc_tax_glyph)
        payroll_taxes = plot.add_glyph(source, payroll_taxes_glyph)
        corporate_income_taxes = plot.add_glyph(source, corporate_income_taxes_glyph)
        excise_taxes = plot.add_glyph(source, excise_taxes_glyph)
        estate_gift_taxes = plot.add_glyph(source, estate_and_gift_taxes_glyph)
        customs_duties = plot.add_glyph(source, customs_duties_glyph)
        misc_receipts = plot.add_glyph(source, misc_receipts_glyph)
        # Add the tools
        tooltips = [
            ("Total", "@total"),
            ("Year", "@year"),
        ]
        ind_inc_tax_hover_tool = HoverTool(renderers=[ind_inc_tax],
                                           tooltips=tooltips + [("Individual Income Taxes", "@individualincometaxes")])
        payroll_taxes_hover_tool = HoverTool(renderers=[payroll_taxes],
                                             tooltips=tooltips + [("Payroll Taxes", "@payrolltaxes")])
        corporate_income_taxes_hover_tool = HoverTool(renderers=[corporate_income_taxes],
                                                      tooltips=tooltips + [
                                                          ("Corporate Income Taxes", "@corporateincometaxes")])
        excise_taxes_hover_tool = HoverTool(renderers=[excise_taxes],
                                            tooltips=tooltips + [("Excise Taxes", "@excisetaxes")])
        estate_gift_taxes_hover_tool = HoverTool(renderers=[estate_gift_taxes],
                                                 tooltips=tooltips + [("Estate/Gift Taxes", "@estateandgifttaxes")])
        customs_duties_hover_tool = HoverTool(renderers=[customs_duties],
                                              tooltips=tooltips + [("Customs Duties", "@customsduties")])
        misc_receipts_hover_tool = HoverTool(renderers=[misc_receipts],
                                             tooltips=tooltips + [("Misc Receipts", "@miscellaneousreceipts")])
        select_tool = BoxSelectTool(renderers=[ind_inc_tax,
                                               payroll_taxes,
                                               corporate_income_taxes,
                                               excise_taxes,
                                               estate_gift_taxes,
                                               customs_duties,
                                               misc_receipts],
                                    dimensions='width')
        plot.add_tools(ind_inc_tax_hover_tool,
                       payroll_taxes_hover_tool,
                       corporate_income_taxes_hover_tool,
                       excise_taxes_hover_tool,
                       estate_gift_taxes_hover_tool,
                       customs_duties_hover_tool,
                       misc_receipts_hover_tool,
                       select_tool)
        layout = Column(plot, data_table)
        doc = Document()
        doc.add_root(layout)
        doc.validate()

        with open(html_output_filename, "w") as f:
            f.write(file_html(doc, INLINE, "CBO Revenues"))
        print("Wrote %s" % html_output_filename)
        view(html_output_filename)

    def bokeh_scatter(self,
                      x_data,
                      y_data,
                      x_label,
                      y_label,
                      point_shape,
                      output_html_filename,
                      size=10,
                      alpha=0.8,
                      color='black'):

        p = figure(x_axis_label=x_label, y_axis_label=y_label)
        if point_shape == 'triangle':
            p.triangle(x_data, y_data, color=color, size=size, alpha=alpha)
        elif point_shape == 'square':
            p.square(x_data, y_data, color=color, size=size, alpha=alpha)
        elif point_shape == 'circle':
            p.circle(x_data, y_data, color=color, size=size, alpha=alpha)
        output_file(output_html_filename)
        show(p)

    def bokeh_double_single_scatter(self,
                                    x_data_1,
                                    y_data_1,
                                    x_data_2,
                                    y_data_2,
                                    x_label,
                                    y_label,
                                    point_shape_1,
                                    point_shape_2,
                                    output_html_filename,
                                    colors=['blue', 'red'],
                                    sizes=[10, 10],
                                    alphas=[0.8, 0.8]):

        p = figure(x_axis_label=x_label, y_axis_label=y_label)
        if point_shape_1 == 'triangle':
            p.triangle(x_data_1, y_data_1, colors[0], sizes[0], alpha[0])
        elif point_shape_1 == 'square':
            p.square(x_data_1, y_data_1, colors[0], sizes[0], alphas[0])
        elif point_shape_1 == 'circle':
            p.circle(x_data_1, y_data_1, colors[0], sizes[0], alphas[0])

        if point_shape_2 == 'triangle':
            p.triangle(x_data_2, y_data_2, colors[1], sizes[1], alphas[1])
        elif point_shape_2 == 'square':
            p.square(x_data_2, y_data_2, colors[1], sizes[1], alphas[1])
        elif point_shape_2 == 'circle':
            p.circle(x_data_2, y_data_2, colors[1], sizes[1], alphas[1])

        output_file(output_html_filename)
        show(p)

    def bokeh_multi_single_scatter(self,
                                   x_data_list,
                                   y_data_list,
                                   x_label,
                                   y_label,
                                   output_html_filename,
                                   colors_list,
                                   sizes_list,
                                   alphas_list):

        p = figure(x_axis_label=x_label, y_axis_label=y_label)
        if len(x_data_list) != len(y_data_list):
            ExtendBokeh.LOGGER.error("data lists cannot be different sizes!")
        else:
            data_list_len = len(x_data_list)
            i = 0
            while i < data_list_len:
                x_data = x_data_list[i]
                y_data = y_data_list[i]
                color = colors_list[i]
                size = sizes_list[i]
                alpha = alphas_list[i]
                p.circle(x_data, y_data, color=color, size=size, alpha=alpha)
                i += 1
            output_file(output_html_filename)
            show(p)

    def bokeh_timeseries_line_plot(self,
                                   x_data,
                                   y_data,
                                   x_label,
                                   y_label,
                                   output_html_filename,
                                   incl_data_points=False):

        """if X axis on a bokeh plot is a time series, then x-acis can be datetime objects
        to specify this in figure() function call, use x-axis type = 'datetime'.
        USE THIS FUNCTION TO PLOT A STOCK PRICE."""

        p = figure(x_axis_type='datetime', x_axis_label=x_label, y_axis_label=y_label)
        p.line(x_data, y_data)
        if incl_data_points:
            p.circle(x_data, y_data, fill_color='blue', size=4)
        output_file(output_html_filename)
        show(p)

    def bokeh_scatter_plot_dataframe(self,
                                     df,
                                     group_by_col_name,
                                     x_data_col,
                                     y_data_col,
                                     output_html_filename,
                                     dot_size=10):
        """ Given a dataframe, plot 2 columns (x, y) grouped by a 3rd columns.
        3rd column must represent actual colors. This method is very restrictive,
        because of the group_by column."""

        p = figure(x_axis_label=x_data_col, y_axis_label=y_data_col)
        p.circle(df[x_data_col], df[y_data_col], color=df[group_by_col_name], size=dot_size)
        output_file(output_html_filename)

    def bokeh_scatter_groupby(self,
                              df,
                              color_col,
                              x_data_col,
                              y_data_col,
                              output_html_filename,
                              dot_size=8):

        p = figure(x_axis_label=x_data_col, y_axis_label=y_data_col)
        source = ColumnDataSource(df)
        p.circle(x_data_col, y_data_col, source=source, color=color_col, size=dot_size)
        output_file(output_html_filename)
        show(p)

    def generate_color_mapper(self,
                              factors_list,
                              color_palette_list):

        # factors list are unique values of a dataframe column to group by
        # for example, country_of_origin = ['Europe', 'Asia', 'US']
        #              color_palette_list = ['red', 'green', 'blue']
        return (CategoricalColorMapper(factors=factors_list,
                                       palette=color_palette_list))

    def bokeh_scatter_groupby_colormapper(self,
                                          df,
                                          x_data_col,
                                          y_data_col,
                                          group_by_col,
                                          color_mapper_obj,
                                          output_html_filename):

        p = figure(x_axis_label=x_data_col, y_axis_label=y_data_col)
        source = ColumnDataSource(df)
        p.circle(x_data_col,
                 y_data_col,
                 source=source,
                 color=dict(field=group_by_col,
                            transform=color_mapper_obj),
                 legend_label=group_by_col)
        output_file(output_html_filename)
        show(p)

    def bokeh_multi_plot(self,
                         df,
                         x_data_col_list,
                         y_data_col_list,
                         output_html_filename,
                         orientation='row'):

        source = ColumnDataSource(df)
        len_x_data_col_list = len(x_data_col_list)
        len_y_data_col_list = len(y_data_col_list)
        if len_x_data_col_list != len_y_data_col_list:
            ExtendBokeh.LOGGER.error("list of x_data must be equal in length to y_data!")
        else:
            i = 0
            figure_list = []
            while i < len(len_x_data_col_list):
                p1 = figure(x_axis_label=x_data_col_list[i],
                            y_axis_label=y_data_col_list[i])
                p1.circle(x_data_col_list[i], y_data_col_list[i], source=source)
                # optional legend, and attributes.
                p1.legend.location = 'bottom_left'
                p1.legend.background_fill_color = 'lightgray'
                figure_list.append(p1)
            if orientation == 'row':
                layout = row(figure_list)
            elif orientation == 'col':
                layout = column(figure_list)
            output_file(output_html_filename)
            show(layout)

    def bokeh_layout_plots_in_grid(self,
                                   num_plots_per_row,
                                   list_of_plots,
                                   output_html_filename):

        row = []
        grid_of_plots = []
        for plot in list_of_plots:
            row.append(plot)
            if len(row) == num_plots_per_row:
                grid_of_plots.append(row)
                row = []
        grid_of_plots.append(row)
        layout = gridplot(grid_of_plots)
        output_file(output_html_filename)
        show(layout)
        return (grid_of_plots)

    def bokeh_layout_plots_in_tabs(self,
                                   title_plot_dict,
                                   output_html_filename):

        list_of_tabs = []
        for plot_title in sorted(set(title_plot_dict.keys())):
            tab = Panel(child=title_plot_dict[plot_title], title=plot_title)
            list_of_tabs.append(tab)
        layout = Tabs(tabs=list_of_tabs)
        output_file(output_html_filename)
        show(layout)

    # linking axes on different charts in a grid layout. Explore how we can use this.

    def bokeh_add_hover_tool(self,
                             label,
                             source_column,
                             figure_obj,
                             output_html_filename):

        # both label and source_column are string
        # source column, since extracting from ColumnDataSource object, is prefixed with '@'
        hover = HoverTool(tooltips=[(label, source_column)])  # i.e. label = 'Country' source_column = '@Country'
        figure_obj.add_tools(output_html_filename)
        output_file('hover.html')
        show(figure_obj)

    @staticmethod
    def bokeh_co_earnings_today_datatable(dataframe,
                                          meta_dict=None):

        source = ColumnDataSource(dataframe)
        # template_orig = """
        #        <div style="font-weight:bold; background:<%=headline_color%>;
        #        word-wrap:break-word; overflow-wrap:break-word; white-space:normal;">
        #        <%= value%></div>
        #        """
        """Reference for the below html: https://stackoverflow.com/questions/2015667/
        how-do-i-set-span-background-color-so-it-colors-the-background-throughout-the-li
        Also see: https://stackoverflow.com/questions/10217915/span-inside-div-prevents-text-overflowellipsis
        """
        template = """
                    <div style="font-weight:bold; overflow:hidden; text-overflow:ellipsis; max-width:1000px; background-color:<%=headline_color%>;>
                    <span style="overflow:hidden; text-overflow:ellipsis; max-width:1000px"; href="#" data-toggle="tooltip" title="<%= value %>"><%= value %>
                    </span></div >
                    """
        html_formatter = HTMLTemplateFormatter(template=template)
        columns = [TableColumn(field=col_nm, title=col_nm, width=1000) for col_nm in dataframe.columns]
        # Hack: Headline column width needs to be larger than the others.
        columns[8].width = 2500
        # Hack: Estimated 1 Year Change, EPS needs to be larger than the others.
        columns[4].width = 2500
        # columns[8].formatter = StringFormatter(font_style = "bold")
        columns[8].formatter = html_formatter
        del columns[9]  # remove the colors columns, we don't want to display it.
        data_table = DataTable(source=source, columns=columns, width=1600, height=800)  # , fit_columns=True)

        return data_table

    @staticmethod
    def bokeh_histogram_overlay_normal(data,
                                       titles=["Px Returns Histogram",
                                               "Px Returns CDF"]):

        hs = HackerStats()
        mu = np.mean(data)
        sigma = np.std(data)
        # ecdf_obj = hs.plot_simulation_ecdf(data = data, x_label = 'Px Ret', y_label='freq', title='Normal')
        ecdf_obj = ECDF(data=data, percentiles=[0.3, 5.0, 32.0, 68.0, 95.0, 99.7])
        ecdf_x = ecdf_obj.get_x_data()
        ecdf_y = ecdf_obj.get_y_data()
        ptiles = ecdf_obj.get_data_ptiles()
        bins = int(np.ceil(hs.get_num_bins_hist(len(data))))
        ExtendBokeh.LOGGER.info("ExtendBokeh.bokeh_histogram_overlay_normal(): ptiles are %s", str(ptiles))
        ExtendBokeh.LOGGER.info("ExtendBokeh.bokeh_histogram_overlay_normal(): number of bins calculated is %s",
                                str(bins))
        hist, edges = np.histogram(data, bins=bins, density=True)
        x = np.linspace(stats.norm.ppf(0.001, loc=mu, scale=sigma),
                        stats.norm.ppf(0.999, loc=mu, scale=sigma), 1000)
        pdf = stats.norm.pdf(x=x, loc=mu, scale=sigma)
        cdf = stats.norm.cdf(x=x, loc=mu, scale=sigma)
        kde = stats.gaussian_kde(data, bw_method='scott')
        kde_pdf = kde.pdf(x)
        # pdf = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
        # cdf = (1 + sp.special.erf((x - mu) / np.sqrt(2 * sigma ** 2))) / 2
        p_hist = make_histogram_plot(titles[0] + " vs. Normal Distribution PDF",
                                     "Avg. Period Return=" +
                                     str("{:.4f}".format(mu * 100.0)) + ", Return Variance=" + str(
                                         "{:.4f}".format(sigma * 100.0)),
                                     hist, edges, x, pdf, cdf, kde_pdf, mu, sigma)
        p_cdf = make_cdf_plot(titles[1] + " vs. Normal Distribution CDF",
                              "Avg. Period Return=" +
                              str("{:.4f}".format(mu * 100.0)) + ", Return Variance=" + str(
                                  "{:.4f}".format(sigma * 100.0)),
                              x, cdf, ecdf_x, ecdf_y, mu, sigma, ptiles)
        return p_hist, p_cdf

    @staticmethod
    def bokeh_histogram_overlay_lognormal(data):

        hs = HackerStats()
        mu = np.mean(data)
        sigma = np.std(data)
        # ecdf_obj = hs.plot_simulation_ecdf(data=data, x_label='Px Ret', y_label='freq', title='LogNormal')
        ecdf_obj = ECDF(data=data)
        ecdf_x = ecdf_obj.get_x_data()
        ecdf_y = ecdf_obj.get_y_data()

        bins = int(np.ceil(hs.get_num_bins_hist(len(data))))
        hist, edges = np.histogram(data, bins=bins, density=True)
        x = np.linspace(np.min(data), np.max(data), num=len(data) * 10)
        # x = np.linspace(0.0001, 8.0, 1000)
        pdf = 1 / (x * sigma * np.sqrt(2 * np.pi)) * np.exp(-(np.log(x) - mu) ** 2 / (2 * sigma ** 2))
        cdf = (1 + sp.special.erf((np.log(x) - mu) / (np.sqrt(2) * sigma))) / 2
        p_lognormal = make_histogram_plot("Log Normal Distribution (μ=0, σ=0.5)",
                                          hist, edges, x, pdf, cdf, kde_pdf=None)
        return p_lognormal

    def bokeh_histogram_overlay_gammma(self,
                                       data):

        hs = HackerStats()
        k = 7.5  # k is the shape of the gamma dist
        theta = 1.0  # theta is the scale of the gamma dist
        theo_gamma_dist = hs.sample_gamma_dist(k=k, theta=theta, size=len(data) * 10)
        bins = int(np.ceil(hs.get_num_bins_hist(len(data))))
        hist, edges = np.histogram(data, density=True, bins=bins)
        x = np.linspace(np.min(data), np.max(data), num=len(data) * 10)
        # x = np.linspace(0.0001, 20.0, 1000)
        pdf = x ** (k - 1) * np.exp(-x / theta) / (theta ** k * sp.special.gamma(k))
        cdf = sp.special.gammainc(k, x / theta)
        p_gamma = make_histogram_plot("Gamma Distribution (k=7.5, θ=1)",
                                      hist, edges, x, pdf, cdf, kde_pdf=None)
        return p_gamma

    def bokeh_histogram_overlay_weibull(self,
                                        data):

        hs = HackerStats()
        lam, k = 1, 1.25
        theo_weibull_dist = hs.sample_weibull_dist(lam=lam, k=k, size=len(data) * 10)
        bins = int(np.ceil(hs.get_num_bins_hist(len(data))))
        hist, edges = np.histogram(data, density=True, bins=bins)
        x = np.linspace(np.min(data), np.max(data), num=len(data) * 10)
        # x = np.linspace(0.0001, 8, 1000)
        pdf = (k / lam) * (x / lam) ** (k - 1) * np.exp(-(x / lam) ** k)
        cdf = 1 - np.exp(-(x / lam) ** k)
        p_weibull = make_histogram_plot("Weibull Distribution (λ=1, k=1.25)",
                                        hist, edges, x, pdf, cdf, kde_pdf=None)
        return p_weibull

    @staticmethod
    def bokeh_create_mean_var_spans(data,
                                    ticker,
                                    benchmark_ticker=None,
                                    freq='D',
                                    rolling_window_size=90,
                                    var_bandwidth=3.0,
                                    color=('red', 'green')):

        data['row_cnt'] = data.reset_index().index.values
        rolling_stat_list = list(filter(lambda col_nm: (str(rolling_window_size) + freq in col_nm) is True,
                                        data.columns.values))
        rolling_mean_stat = list(filter(lambda col_nm: ('mean' in col_nm) is True, rolling_stat_list))[0]
        rolling_std_stat = list(filter(lambda col_nm: ('std' in col_nm) is True, rolling_stat_list))[0]
        data['lower'] = data[rolling_mean_stat] - data[rolling_std_stat] * var_bandwidth
        data['upper'] = data[rolling_mean_stat] + data[rolling_std_stat] * var_bandwidth
        if benchmark_ticker is not None:
            px_ret_type_list = list(filter(lambda col_nm: ('_excess_rets' in col_nm) is True, data.columns.values))
        else:
            px_ret_type_list = list(filter(lambda col_nm: ('_px_rets' in col_nm) is True, data.columns.values))
        lower_breach = data[data.lower > data[px_ret_type_list[0]]]
        upper_breach = data[data.upper < data[px_ret_type_list[0]]]
        return_tuples_list = [(color[0], lower_breach), (color[1], upper_breach)]

        return return_tuples_list

    @staticmethod
    def bokeh_px_line_plot(data,
                           ticker,
                           benchmark_px_series=None,
                           benchmark_ticker=None,
                           title=['Px Chart'],
                           subtitle=[''],
                           type_list=['adjClose'],
                           color_list=['navy', 'green', 'blue', 'limegreen', 'black', 'magenta'],
                           spans_list=None,
                           which_axis_list=None):

        x_axis = pd.to_datetime(data.index)
        p = figure(plot_width=600, plot_height=400, x_axis_type="datetime")
        p.add_layout(Title(text=subtitle[0], text_font_style="italic"), place='above')
        p.add_layout(Title(text=title[0], text_font_size="14pt"), place='above')
        num_lines_to_plot = len(type_list)
        if which_axis_list is None:
            which_axis_list = [0 for x in range(num_lines_to_plot)]
        px_type_color_axis_list = list(zip(color_list[0:len(type_list)], type_list, which_axis_list))
        for color, px_type, which_axis in px_type_color_axis_list:
            if which_axis is 0:
                p.y_range = Range1d(start=data[px_type].min(), end=data[px_type].max())
                p.line(x_axis, data[px_type], color=color, alpha=0.5, legend_label=ticker)
            else:
                p.extra_y_ranges = {benchmark_ticker + '_Px': Range1d(start=benchmark_px_series.min().values[0],
                                                                      end=benchmark_px_series.max().values[0])}
                p.add_layout(LinearAxis(y_range_name=benchmark_ticker + '_Px'), 'right')
                p.line(x_axis, benchmark_px_series.squeeze(), color='black', y_range_name=benchmark_ticker + '_Px',
                       legend_label=benchmark_ticker)
        if spans_list is not None:
            for span_tuple in spans_list:
                span_obj_series = span_tuple[1].apply(lambda x: Span(location=x.name,
                                                                     dimension='height',
                                                                     line_color=span_tuple[0],
                                                                     line_dash='dashed',
                                                                     line_width=1), 1)
                span_obj_series.apply(lambda x: p.add_layout(x), 1)
        p.legend.location = "top_left"
        p.legend.background_fill_color = "#fefefe"
        return p

    @staticmethod
    def bokeh_ed_ir_rolling_ticks_correl(data,
                                         diff_types_to_correlate,
                                         type_list,
                                         title=['ED/IR Rolling Cum. Sum vs. Correl'],
                                         subtitle=[''],
                                         rolling_window_size=30,
                                         correl_filter=None):

        data["x_coord"] = pd.to_datetime(data.index)
        p_scat_1 = figure(plot_width=600, plot_height=400)
        p_scat_1.add_layout(Title(text=subtitle[0], text_font_style="italic"), place='above')
        p_scat_1.add_layout(Title(text=title[0], text_font_size="14pt"), place='above')

        p_scat_2 = figure(plot_width=600, plot_height=400)
        p_scat_2.add_layout(Title(text=subtitle[1], text_font_style="italic"), place='above')
        p_scat_2.add_layout(Title(text=title[1], text_font_size="14pt"), place='above')

        p_scat_3 = figure(plot_width=600, plot_height=400)
        p_scat_3.add_layout(Title(text=subtitle[2], text_font_style="italic"), place='above')
        p_scat_3.add_layout(Title(text=title[2], text_font_size="14pt"), place='above')

        p_line_correl = figure(plot_width=600, plot_height=400, x_axis_type='datetime')
        p_line_correl.add_layout(Title(text=subtitle[3], text_font_style="italic"), place='above')
        p_line_correl.add_layout(Title(text=title[3], text_font_size="14pt"), place='above')
        """
        type_list = ['rolling_reversion_trade_pnl',
                     'fwd_looking_rolling_reversion_trade_pnl',
                     'SettleLastTradeSelect',
                     'os_sl_lagged_corr_series']
        
        """
        scatter_source = ColumnDataSource(data)
        p_line_correl.line(x="x_coord", y="os_sl_lagged_corr_series", color='blue', alpha=0.5, source=scatter_source,
                           legend_label='Correlation')
        filtered_data = data
        if correl_filter is not None:
            filtered_data = data[correl_filter[diff_types_to_correlate][2](
                correl_filter[diff_types_to_correlate][0][0](data[type_list[3]],
                                                             correl_filter[diff_types_to_correlate][0][1]),
                correl_filter[diff_types_to_correlate][1][0](data[type_list[3]],
                                                             correl_filter[diff_types_to_correlate][1][1]))]
            scatter_source = ColumnDataSource(filtered_data)
        # scatter plots
        p_scat_1.scatter(x=type_list[3], y=type_list[0], line_color=None, size=5,
                         source=scatter_source, legend_label=type_list[0])
        p_scat_2.scatter(x=type_list[3], y=type_list[1], line_color=None, size=5,
                         source=scatter_source, legend_label=type_list[1])
        p_scat_3.scatter(x=type_list[3], y=type_list[2], line_color=None, size=5,
                         source=scatter_source, legend_label=type_list[2])
        # regression line
        regression_1 = np.polyfit(filtered_data[type_list[3]],
                                  filtered_data[type_list[0]], 1)
        regression_2 = np.polyfit(filtered_data[type_list[3]],
                                  filtered_data[type_list[1]], 1)
        regression_3 = np.polyfit(filtered_data[type_list[3]],
                                  filtered_data[type_list[2]], 1)
        # min/max x-axis
        min_val = filtered_data[type_list[3]].min()
        max_val = filtered_data[type_list[3]].max()
        # regression lines
        r_x = np.linspace(start=min_val, stop=max_val, num=len(filtered_data))
        r_y_1 = r_x * regression_1[0] + regression_1[1]
        r_y_2 = r_x * regression_2[0] + regression_2[1]
        r_y_3 = r_x * regression_3[0] + regression_3[1]
        # legend and plot regression line
        p_scat_1.line(x=r_x, y=r_y_1, color='red')
        p_scat_1.legend.location = "top_left"
        p_scat_1.legend.background_fill_color = "#fefefe"
        # legend and plot regression line
        p_scat_2.line(x=r_x, y=r_y_2, color='red')
        p_scat_2.legend.location = "top_left"
        p_scat_2.legend.background_fill_color = "#fefefe"
        # legend and plot regression line
        p_scat_3.line(x=r_x, y=r_y_3, color='red')
        p_scat_3.legend.location = "top_left"
        p_scat_3.legend.background_fill_color = "#fefefe"
        # legend and plot correlation line
        p_line_correl.legend.location = "top_left"
        p_line_correl.legend.background_fill_color = "#fefefe"
        # return the figures
        return p_scat_1, p_scat_2, p_scat_3, p_line_correl

    @staticmethod
    def bokeh_rolling_pxret_skew(data,
                                 freq='D',
                                 title=["Rolling Returns vs. Rolling Skew"],
                                 subtitle=[''],
                                 type_list=None,
                                 color_list=['blue', 'magenta'],
                                 band_width=3.0,
                                 scatter=True,
                                 rolling_window_size=90,
                                 skew_filter=None):

        data["x_coord"] = pd.to_datetime(data.index)
        p_line = figure(plot_width=600, plot_height=400, x_axis_type='datetime')
        p_scat = figure(plot_width=600, plot_height=400)
        p_line.add_layout(Title(text=subtitle[0], text_font_style="italic"), place='above')
        p_line.add_layout(Title(text=title[0], text_font_size="14pt"), place='above')
        p_scat.add_layout(Title(text=subtitle[0], text_font_style="italic"), place='above')
        p_scat.add_layout(Title(text=title[0], text_font_size="14pt"), place='above')

        px_type_color_dict = dict(zip(color_list[0:len(type_list)], type_list))
        data['row_cnt'] = data.reset_index().index.values

        rolling_stat_list = list(filter(lambda col_nm: (str(rolling_window_size) + freq in col_nm) is True,
                                        data.columns.values))
        rolling_mean_stat = list(filter(lambda col_nm: ('mean' in col_nm) is True, rolling_stat_list))[0]
        rolling_std_stat = list(filter(lambda col_nm: ('std' in col_nm) is True, rolling_stat_list))[0]
        rolling_skew_stat = list(filter(lambda col_nm: ('skew' in col_nm) is True, rolling_stat_list))[0]
        rolling_kurt_stat = list(filter(lambda col_nm: ('kurtosis' in col_nm) is True, rolling_stat_list))[0]
        rolling_sem_stat = list(filter(lambda col_nm: ('sem' in col_nm) is True, rolling_stat_list))[0]

        source = ColumnDataSource(data)
        series_cnt = 0
        for color, px_ret_type in px_type_color_dict.items():
            ecdf_obj = ECDF(data=data[px_ret_type], percentiles=[0.3, 5.0, 32.0, 68.0, 95.0, 99.7])
            mu = ecdf_obj.get_mu()
            sigma = ecdf_obj.get_sigma()
            ExtendBokeh.LOGGER.info("ExtendBokeh.bokeh_rolling_pxret_skew(): plotting %s in line color %s",
                                    px_ret_type, color)
            if series_cnt == 0:
                p_line.y_range = Range1d(data[px_ret_type].min(), data[px_ret_type].max())
                p_line.line(x="x_coord", y=px_ret_type, color=color, alpha=0.5, source=source, legend_label=px_ret_type)
            else:
                p_line.extra_y_ranges = {"foo": DataRange1d(data[px_ret_type].min(), data[px_ret_type].max())}
                p_line.line(x="x_coord", y=px_ret_type, color=color, alpha=0.5, source=source, legend_label=px_ret_type,
                            y_range_name="foo")
                p_line.add_layout(LinearAxis(y_range_name="foo"), 'right')

            series_cnt += 1
        if scatter is True:
            scatter_source = source
            if skew_filter is not None:
                filtered_data = data[data[type_list[1]] < skew_filter[0]]
                filtered_data = pd.concat([filtered_data, data[data[type_list[1]] > skew_filter[1]]])
                scatter_source = ColumnDataSource(filtered_data)
            p_scat.circle(x=type_list[1], y=type_list[0], line_color=None, size=5,
                          source=scatter_source, legend_label=type_list[0])
            # regression line
            regression_failed = False
            try:
                data.dropna(inplace=True)
                regression = np.polyfit(data[type_list[1]][rolling_window_size:],
                                        data[type_list[0]][rolling_window_size:],
                                        1)
            except np.linalg.LinAlgError as lae:
                ExtendBokeh.LOGGER.error(
                    "extend_bokeh.bokeh_rolling_pxret_skew(): error during np.polyfit running regression: %s", str(lae))
                regression_failed = True
            if regression_failed is False:
                min_val = data[type_list[1]].min()
                max_val = data[type_list[1]].max()
                # r_x, r_y = zip(*((i, i*regression[0] + regression[1]) for i in range(min_val, max_val)))
                r_x = np.linspace(start=min_val, stop=max_val, num=len(data))
                r_y = r_x * regression[0] + regression[1]
                p_scat.line(x=r_x, y=r_y, color='red')

                p_scat.legend.location = "top_left"
                p_scat.legend.background_fill_color = "#fefefe"
            elif regression_failed is True:
                p_line.legend.location = "top_left"
                p_line.legend.background_fill_color = "#fefefe"
                return p_line
        p_line.legend.location = "top_left"
        p_line.legend.background_fill_color = "#fefefe"
        return p_line, p_scat

    @staticmethod
    def bokeh_px_returns_plot(data,
                              freq='D',
                              title=['Px Returns Chart'],
                              subtitle=[''],
                              type_list=None,
                              color_list=['navy', 'green', 'blue', 'limegreen', 'black', 'magenta'],
                              band_width=3.0,
                              scatter=False,
                              rolling_window_size=90):  # options are 30, 60, 90, 120, 180, 270

        x_axis = pd.to_datetime(data.index)
        p = figure(plot_width=600, plot_height=400, x_axis_type='datetime')
        p.add_layout(Title(text=subtitle[0], text_font_style="italic"), place='above')
        p.add_layout(Title(text=title[0], text_font_size="14pt"), place='above')
        px_type_color_dict = dict(zip(color_list[0:len(type_list)], type_list))
        data['row_cnt'] = data.reset_index().index.values
        rolling_stat_list = list(filter(lambda col_nm: (str(rolling_window_size) + freq in col_nm) is True,
                                        data.columns.values))
        rolling_mean_stat = list(filter(lambda col_nm: ('mean' in col_nm) is True, rolling_stat_list))[0]
        rolling_std_stat = list(filter(lambda col_nm: ('std' in col_nm) is True, rolling_stat_list))[0]
        rolling_skew_stat = list(filter(lambda col_nm: ('skew' in col_nm) is True, rolling_stat_list))[0]
        rolling_kurt_stat = list(filter(lambda col_nm: ('kurtosis' in col_nm) is True, rolling_stat_list))[0]
        rolling_sem_stat = list(filter(lambda col_nm: ('sem' in col_nm) is True, rolling_stat_list))[0]

        data['lower'] = data[rolling_mean_stat] - data[rolling_std_stat] * band_width
        data['upper'] = data[rolling_mean_stat] + data[rolling_std_stat] * band_width
        source = ColumnDataSource(data)

        for color, px_ret_type in px_type_color_dict.items():
            ecdf_obj = ECDF(data=data[px_ret_type], percentiles=[0.3, 5.0, 32.0, 68.0, 95.0, 99.7])
            mu = ecdf_obj.get_mu()
            sigma = ecdf_obj.get_sigma()
            if (not scatter):
                ExtendBokeh.LOGGER.info("ExtendBokeh.bokeh_px_returns_plot(): plotting %s in line color %s",
                                        px_ret_type, color)
                p.line(x_axis, data[px_ret_type], color=color, alpha=0.5, legend_label=px_ret_type)
            else:
                p.scatter(x='date', y=px_ret_type, line_color=None, fill_color=color,
                          fill_alpha=0.7, size=5, source=source, legend_label=px_ret_type)
            band = Band(base='date', lower='lower', upper='upper', level='underlay',
                        fill_alpha=1.0, line_width=1, line_color='black', source=source)
            p.add_layout(band)

        minus_three_sigma_box = BoxAnnotation(bottom=mu - 3 * sigma, top=mu - 2 * sigma, fill_alpha=0.2,
                                              fill_color='red')
        minus_two_sigma_box = BoxAnnotation(bottom=mu - 2 * sigma, top=mu - sigma, fill_alpha=0.2,
                                            fill_color='lightcoral')
        minus_one_sigma_box = BoxAnnotation(bottom=mu - sigma, top=mu, fill_alpha=0.2, fill_color='lightsalmon')
        plus_three_sigma_box = BoxAnnotation(top=mu + 3 * sigma, bottom=mu + 2 * sigma, fill_alpha=0.2,
                                             fill_color='limegreen')
        plus_two_sigma_box = BoxAnnotation(top=mu + 2 * sigma, bottom=mu + sigma, fill_alpha=0.2,
                                           fill_color='green')
        plus_one_sigma_box = BoxAnnotation(top=mu + sigma, bottom=mu, fill_alpha=0.2, fill_color='palegreen')

        p.add_layout(minus_three_sigma_box)
        p.add_layout(minus_two_sigma_box)
        p.add_layout(minus_one_sigma_box)
        p.add_layout(plus_three_sigma_box)
        p.add_layout(plus_two_sigma_box)
        p.add_layout(plus_one_sigma_box)

        p.legend.location = "bottom_right"
        p.legend.background_fill_color = "#fefefe"

        return p

    @staticmethod
    def bokeh_bar_plot(data_df, data_df_out_csv, html_output_file, html_output_file_title):
        output_file(html_output_file, title=html_output_file_title)
        data_df['Year'] = pd.DatetimeIndex(data_df.Date).year
        data_df['Month'] = pd.DatetimeIndex(data_df.Date).month
        data_df['Day'] = pd.DatetimeIndex(data_df.Date).day
        data_df.Year = data_df.Year.astype(str)
        group = data_df.groupby(by=['Year', 'Ticker'])
        data_df_rolling_returns_sharpe = group['RollingReturns']. \
            agg([lambda x: x.mean() / x.std() * np.sqrt(x.size), np.mean, np.std, np.size]). \
            rename(columns={'<lambda_0>': 'SharpeRatio',
                            'mean': 'AverageRollingReturns',
                            'std': 'StdDevRollingReturns',
                            'size': 'SampleSizeReturns'})
        index_cmap = factor_cmap('Year_Ticker', palette=Spectral5, factors=sorted(data_df.Year.unique()), end=1)
        p = figure(plot_width=900, plot_height=600, title=html_output_file_title,
                   x_range=group, toolbar_location=None, tooltips=[("SharpeRatio", "@SharpeRatio"),
                                                                   ("Year, Ticker", "@Year_Ticker"),
                                                                   ("Mean Rolling Return", "@AverageRollingReturns"),
                                                                   ("Std Dev Rolling Return", "@StdDevRollingReturns"),
                                                                   ("Sample Size", "@SampleSizeReturns")
                                                                   ])
        p.vbar(x='Year_Ticker', top='SharpeRatio', width=1, source=data_df_rolling_returns_sharpe, line_color="white",
               fill_color=index_cmap, )
        p.x_range.range_padding = 0.05
        p.xgrid.grid_line_color = None
        p.xaxis.axis_label = "Ticker grouped by Year"
        p.xaxis.major_label_orientation = 1.2
        p.outline_line_color = None
        show(p)

    @staticmethod
    def bokeh_bar_plot_single_group(data_df, data_df_out_csv, html_output_file, html_output_file_title):
        output_file(html_output_file, title=html_output_file_title)
        data_df.Ticker = data_df.Ticker.astype(str)
        group = data_df.groupby('Ticker')
        data_df_rolling_returns_sharpe = group['RollingReturns']. \
            agg([lambda x: x.mean() / x.std() * np.sqrt(x.size), np.mean, np.std, np.size]). \
            rename(columns={'<lambda_0>': 'SharpeRatio',
                            'mean': 'AverageRollingReturns',
                            'std': 'StdDevRollingReturns',
                            'size': 'SampleSizeReturns'})
        index_cmap = factor_cmap('Ticker', palette=Spectral5, factors=sorted(data_df.Ticker.unique()))
        p = figure(plot_height=600, plot_width=900, x_range=group, title=html_output_file_title,
                   toolbar_location=None, tooltips=[("SharpeRatio", "@SharpeRatio"),
                                                    ("Ticker", "@Ticker"),
                                                    ("Mean Rolling Return", "@AverageRollingReturns"),
                                                    ("Std Dev Rolling Return", "@StdDevRollingReturns"),
                                                    ("Sample Size", "@SampleSizeReturns")])
        p.vbar(x="Ticker", top='SharpeRatio', width=1, source=data_df_rolling_returns_sharpe, line_color='white',
               fill_color=index_cmap, )
        p.xgrid.grid_line_color = None
        p.xaxis.axis_label = "Skew Values, Binned"
        p.xaxis.major_label_orientation = 1.2
        p.outline_line_color = None
        show(p)

    @staticmethod
    def bokeh_bar_plot_all_bins_by_ticker(data_df, data_df_out_csv, html_output_file, html_output_file_title):
        output_file(html_output_file, title=html_output_file_title)
        c_float = data_df.Bin.unique()
        ticker_unique = data_df.Ticker.unique()
        c_str = [str(a) for a in c_float]
        category_dict = OrderedDict(zip(c_float, c_str))
        the_factors = [(a_bin, a_ticker) for a_bin in list(category_dict.values()) for a_ticker in ticker_unique]
        data_df.Bin = data_df.Bin.astype(str)
        group = data_df.groupby(by=['Bin','Ticker'])
        data_df_rolling_returns_sharpe = group['RollingReturns']. \
            agg([lambda x: x.mean() / x.std() * np.sqrt(x.size), np.mean, np.std, np.size]). \
            rename(columns={'<lambda_0>': 'SharpeRatio',
                            'mean': 'AverageRollingReturns',
                            'std': 'StdDevRollingReturns',
                            'size': 'SampleSizeReturns'})
        index_cmap = factor_cmap('Bin_Ticker', palette=Spectral5, factors=list(category_dict.values()), end=1)
        p = figure(plot_height=600, plot_width=900, x_range=group, title=html_output_file_title,
                   toolbar_location=None, tooltips=[("SharpeRatio", "@SharpeRatio"),
                                                    ("Bin_Ticker", "@Bin_Ticker"),
                                                    ("Mean Rolling Return", "@AverageRollingReturns"),
                                                    ("Std Dev Rolling Return", "@StdDevRollingReturns"),
                                                    ("Sample Size", "@SampleSizeReturns")])
        p.vbar(x="Bin_Ticker", top='SharpeRatio', width=1, source=data_df_rolling_returns_sharpe, line_color='white',
               fill_color=index_cmap, )
        p.x_range.factors = the_factors
        p.x_range.range_padding = 0.05
        p.xgrid.grid_line_color = None
        p.xaxis.axis_label = "Bins, Grouped by Ticker"
        p.xaxis.major_label_orientation = 1.2
        p.outline_line_color = None
        show(p)


    @staticmethod
    def bokeh_bar_plot_all_bins(data_df, data_df_out_csv, html_output_file, html_output_file_title):
        output_file(html_output_file, title=html_output_file_title)
        c_float = data_df.Bin.unique()
        c_str = [str(a) for a in c_float]
        category_dict = OrderedDict(zip(c_float, c_str))
        data_df.Bin = data_df.Bin.astype(str)
        group = data_df.groupby('Bin')
        data_df_rolling_returns_sharpe = group['RollingReturns'].\
            agg([lambda x: x.mean() / x.std() * np.sqrt(x.size), np.mean, np.std, np.size]).\
            rename(columns={'<lambda_0>': 'SharpeRatio',
                            'mean':'AverageRollingReturns',
                            'std': 'StdDevRollingReturns',
                            'size': 'SampleSizeReturns'})
        index_cmap = factor_cmap('Bin', palette=Spectral5, factors=list(category_dict.values()))
        p = figure(plot_height=600, plot_width=900, x_range=group, title=html_output_file_title,
                   toolbar_location=None, tooltips=[("SharpeRatio", "@SharpeRatio"),
                                                    ("Bin", "@Bin"),
                                                    ("Mean Rolling Return", "@AverageRollingReturns"),
                                                    ("Std Dev Rolling Return", "@StdDevRollingReturns"),
                                                    ("Sample Size", "@SampleSizeReturns")])
        p.vbar(x="Bin", top='SharpeRatio', width=1, source=data_df_rolling_returns_sharpe, line_color='white',
              fill_color=index_cmap, )
        p.x_range.factors = list(category_dict.values())
        p.xgrid.grid_line_color = None
        p.xaxis.axis_label = "Ticker, Both hedged with Benchmark and not"
        p.xaxis.axis_label_text_align = 'left'
        p.xaxis.major_label_orientation = 1.2
        p.outline_line_color = None
        show(p)

    @staticmethod
    def show_hist_plots(grid_plots_list,
                        html_output_file,
                        html_output_file_title):

        output_file(html_output_file, title=html_output_file_title)
        gp = gridplot(grid_plots_list, ncols=2, plot_width=600, plot_height=600, toolbar_location='right')
        show(gp)

    @staticmethod
    def save_html(grid_plots_list,
                  html_output_file,
                  html_output_file_title):

        output_file(html_output_file, title=html_output_file_title)
        gp = gridplot(grid_plots_list, ncols=2, plot_width=600, plot_height=600, toolbar_location='right')
        # toolbar_location options are 'above', 'right', 'left', 'below'
        save(gp)

    @staticmethod
    def show_co_earnings_today_data_table(data_table,
                                          html_output_file,
                                          html_output_file_title):

        output_file(html_output_file, title=html_output_file_title)
        show(widgetbox(data_table))

    @staticmethod
    def save_co_earnings_today_data_table(data_table,
                                          html_output_file,
                                          html_output_file_title):

        output_file(html_output_file, title=html_output_file_title)
        save(widgetbox(data_table))


def make_histogram_plot(title,
                        subtitle,
                        hist,
                        edges,
                        x,
                        pdf,
                        cdf,
                        kde_pdf,
                        mu,
                        sigma):
    """ Name: make_histogram_plot
        Function: create a histogram plot with the input title, edges,
                x-axis data, and overlays the PDF and CDF line data.
        Parameters: title: title of chart
                    hist: histogram object, for example, returnd from np.histogram()
                    edges: arrays of x data, edges of bins, returned from np.histogram()
                    x: x-axis data to plot the histogram.
                    pdf: Probability Distribution Function: i.e for normal dist, bell curve.
                    cdf: Cummulative Distribution Function.
    """
    p = figure(tools='', background_fill_color="#fafafa")

    minus_three_sigma_box = BoxAnnotation(left=mu - 3 * sigma, right=mu - 2 * sigma, fill_alpha=0.1, fill_color='red')
    minus_two_sigma_box = BoxAnnotation(left=mu - 2 * sigma, right=mu - sigma, fill_alpha=0.1, fill_color='lightcoral')
    minus_one_sigma_box = BoxAnnotation(left=mu - sigma, right=mu, fill_alpha=0.1, fill_color='lightsalmon')
    plus_three_sigma_box = BoxAnnotation(right=mu + 3 * sigma, left=mu + 2 * sigma, fill_alpha=0.1, fill_color='green')
    plus_two_sigma_box = BoxAnnotation(right=mu + 2 * sigma, left=mu + sigma, fill_alpha=0.1, fill_color='limegreen')
    plus_one_sigma_box = BoxAnnotation(right=mu + sigma, left=mu, fill_alpha=0.1, fill_color='palegreen')
    p.add_layout(minus_three_sigma_box)
    p.add_layout(minus_two_sigma_box)
    p.add_layout(minus_one_sigma_box)
    p.add_layout(plus_three_sigma_box)
    p.add_layout(plus_two_sigma_box)
    p.add_layout(plus_one_sigma_box)
    p.add_layout(Title(text=subtitle, text_font_style="italic"), place='above')
    p.add_layout(Title(text=title, text_font_size="14pt"), place='above')
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
           fill_color="navy", line_color="white", alpha=0.5)
    p.line(x, pdf, line_color="#ff8888", line_width=4, alpha=0.7, legend_label="PDF")
    # p.line(x, cdf, line_color="orange", line_width=2, alpha=0.7, legend="CDF")
    if kde_pdf is not None:
        p.line(x, kde_pdf, line_color="black", line_width=2, alpha=0.7, legend_label='Density')
    p.y_range.start = 0
    p.legend.location = "center_right"
    p.legend.background_fill_color = "#fefefe"
    p.xaxis.axis_label = 'x'
    p.yaxis.axis_label = 'Pr(x)'
    p.grid.grid_line_color = "white"

    return p


def make_cdf_plot(title, subtitle, x, cdf, ecdf_x, ecdf_y, mu, sigma, ptiles):
    # p = figure(title=title, tools='', background_fill_color="#fafafa")
    p = figure(tools='', background_fill_color="#fafafa")
    for ptile in ptiles:
        ptile_span = Span(location=ptile,
                          dimension='height',
                          line_color='red',
                          line_dash='dashed',
                          line_width=3)
        p.add_layout(ptile_span)

    minus_three_sigma_box = BoxAnnotation(left=mu - 3 * sigma, right=mu - 2 * sigma, fill_alpha=0.1, fill_color='red')
    minus_two_sigma_box = BoxAnnotation(left=mu - 2 * sigma, right=mu - sigma, fill_alpha=0.1, fill_color='lightcoral')
    minus_one_sigma_box = BoxAnnotation(left=mu - sigma, right=mu, fill_alpha=0.1, fill_color='lightsalmon')
    plus_three_sigma_box = BoxAnnotation(right=mu + 3 * sigma, left=mu + 2 * sigma, fill_alpha=0.1, fill_color='green')
    plus_two_sigma_box = BoxAnnotation(right=mu + 2 * sigma, left=mu + sigma, fill_alpha=0.1, fill_color='limegreen')
    plus_one_sigma_box = BoxAnnotation(right=mu + sigma, left=mu, fill_alpha=0.1, fill_color='palegreen')
    p.add_layout(minus_three_sigma_box)
    p.add_layout(minus_two_sigma_box)
    p.add_layout(minus_one_sigma_box)
    p.add_layout(plus_three_sigma_box)
    p.add_layout(plus_two_sigma_box)
    p.add_layout(plus_one_sigma_box)
    p.add_layout(Title(text=subtitle, text_font_style="italic"), place='above')
    p.add_layout(Title(text=title, text_font_size="14pt"), place='above')
    # p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
    #       fill_color="navy", line_color="white", alpha=0.5)
    p.line(ecdf_x, ecdf_y, line_color='blue', line_width=2, alpha=0.7, legend_label="ECDF")
    p.line(x, cdf, line_color="orange", line_width=2, alpha=0.7, legend_label="CDF")
    p.y_range.start = 0
    p.legend.location = "center_right"
    p.legend.background_fill_color = "#fefefe"
    p.xaxis.axis_label = 'x'
    p.yaxis.axis_label = 'Pr(x)'
    p.grid.grid_line_color = "white"

    return p


if __name__ == '__main__':
    ex_bokeh = ExtendBokeh()
    return_val = ex_bokeh.bokeh_layout_plots_in_grid(2,
                                                     [2, 3, 4, 5, 6])
    print(return_val)
