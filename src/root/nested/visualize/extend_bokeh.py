from bokeh.plotting import figure, ColumnDataSource
from bokeh.io import output_file, show
from bokeh.layouts import row, column, gridplot
from bokeh.models.widgets import Panel, Tabs
from bokeh.models import HoverTool


from root.nested import get_logger

class ExtendBokeh(object):
    """description of class"""

    def __init__(self, **kwargs):

        self.logger = get_logger()
        return super().__init__(**kwargs)


    def bokeh_scatter(self,
                      x_data,
                      y_data,
                      x_label,
                      y_label,
                      point_shape,
                      output_html_filename,
                      size = 10,
                      alpha = 0.8,
                      color = 'black'):

        p = figure(x_axis_label = x_label, y_axis_label = y_label)
        if point_shape == 'triangle':
            p.triangle(x_data, y_data, color = color, size = size, alpha = alpha)
        elif point_shape == 'square':
            p.square(x_data, y_data, color = color, size = size, alpha=alpha)
        elif point_shape == 'circle':
            p.circle(x_data, y_data, color = color, size = size, alpha = alpha)
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
                                    colors = ['blue', 'red'],
                                    sizes = [10, 10],
                                    alphas = [0.8, 0.8]):

        p = figure(x_axis_label = x_label, y_axis_label = y_label)
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
                                   point_shape_list,
                                   output_html_filename,
                                   colors_list,
                                   sizes_list,
                                   alphas_list):

        p = figure(x_axis_label = x_label, y_axis_label = y_label)
        if len(x_data_list) != len(y_data_list):
            self.logger.error("data lists cannot be different sizes!")
        else:
            data_list_len = len(x_data_list)
            i = 0
            while i < data_list_len:
                x_data = x_data_list[i]
                y_data = y_data_list[i]
                color = colors_list[i]
                size = sizes_list[i]
                alpha = alphas_list[i]
                p.circle(x_data, y_data, color = color, size = size, alpha = alpha)
                i+=1
            output_file(output_html_filename)
            show(p)

    ## if x_axis on a bokeh plot is a time series, then x-axis can be datetime objects
    ## to specify this in figure() function call, use x_axis_type = 'datetime'

    ## if plotting a stock price, use this function
    def bokeh_timeseries_line_plot(self,
                                   x_data,
                                   y_data,
                                   x_label,
                                   y_label,
                                   output_html_filename,
                                   incl_data_points=False):

        p = figure(x_axis_type = 'datetime', x_axis_label = x_label, y_axis_label = y_label)
        p.line(x_data, y_data)
        if incl_data_points:
            p.circle(x_data, y_data, fill_color = 'blue', size = 4)
        output_file(output_html_filename)
        show(p)

    ## given a dataframe, plot two columns (x,y) grouped by a 3rd column
    ## 3rd column must represent actual colors
    ## this method is very restricted by the group_by column
    def bokeh_scatter_plot_dataframe(self,
                                     df,
                                     group_by_col_name,
                                     x_data_col,
                                     y_data_col,
                                     output_html_filename,
                                     dot_size = 10):

        p = figure(x_axis_label = x_data_col, y_axis_label = y_data_col)
        p.circle(df[x_data_col], df[y_data_col], color = df[group_by_col_name], size = dot_size)
        output_file(output_html_filename)
        
    def bokeh_scatter_groupby(self,
                              df,
                              color_col,
                              x_data_col,
                              y_data_col,
                              output_html_filename,
                              dot_size = 8):

        p = figure(x_axis_label = x_data_col, y_axis_label = y_data_col)
        source = ColumnDataSource(df)
        p.circle(x_data_col, y_data_col, source = source, color = color_col, size = dot_size)
        output_file(output_html_filename)
        show(p)

    def generate_color_mapper(self,
                              factors_list,
                              color_palette_list):

        # factors list are unique values of a dataframe column to group by
        # for example, country_of_origin = ['Europe', 'Asia', 'US']
        #              color_palette_list = ['red', 'green', 'blue']
        return (CategoricalColorMapper(factors = factors_list,
                                       palette = color_palette_list))

    def bokeh_scatter_groupby_colormapper(self,
                                          df,
                                          x_data_col,
                                          y_data_col,
                                          group_by_col,
                                          color_mapper_obj,
                                          output_html_filename):

        p = figure(x_axis_label = x_data_col, y_axis_label = y_data_col)
        source = ColumnDataSource(df)
        p.circle(x_data_col, 
                 y_data_col, 
                 source = source, 
                 color = dict(field =group_by_col, 
                              transform=color_mapper_obj), 
                 legend=group_by_col)
        output_file(output_html_filename)
        show(p)

    def bokeh_multi_plot(self,
                         df,
                         x_data_col_list,
                         y_data_col_list,
                         output_html_filename,
                         orientation = 'row'):

        source = ColumnDataSource(df)
        len_x_data_col_list = len(x_data_col_list)
        len_y_data_col_list = len(y_data_col_list)
        if len_x_data_col_list != len_y_data_col_list:
            self.logger.error("list of x_data must be equal in length to y_data!")
        else:
            i = 0
            figure_list = []
            while i < len(len_x_data_col_list):
                p1 = figure(x_axis_label = x_data_col_list[i],
                            y_axis_label = y_data_col_list[i])
                p1.circle(x_data_col_list[i], y_data_col_list[i], source = source, legend=None)
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
            tab = Panel(child = title_plot_dict[plot_title], title = plot_title )
            list_of_tabs.append(tab)
        layout = Tabs(tabs = list_of_tabs)
        output_file(output_html_filename)
        show(layout)

    ## linking axes oon different charts in a grid layout. Explore how we can use this.

    def bokeh_add_hover_tool(self,
                             label,
                             source_column,
                             figure_obj,
                             output_html_filename):

        # both label and source_column are string
        # source column, since extracting from ColumnDataSource object, is prefixed with '@'
        hover = HoverTool(tooltips = [(label, source_column)]) # i.e. label = 'Country' source_column = '@Country'
        figure_obj.add_tools(output_html_filename)
        output_file('hover.html')
        show(figure_obj)




if __name__ == '__main__':

    ex_bokeh = ExtendBokeh()
    return_val = ex_bokeh.bokeh_layout_plots_in_grid(2,
                                                     [2,3,4,5,6])
    print(return_val)





