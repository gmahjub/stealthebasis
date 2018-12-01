from bokeh.io import output_file, show, curdoc
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper, Slider, Select
from bokeh.palettes import Spectral6
from bokeh.layouts import widgetbox, row

import pandas as pd
import numpy as np

class BokehGapminderApp(object):
    """description of class"""

    def __init__(self, **kwargs):

        self.df = pd.read_csv('C:\\Users\\ghazy\\workspace\\data\\datacamp\\gapminder_tidy.csv',
                                        sep = ',',
                                        header = 0,
                                        index_col = 'Year')
        self.slider = None
        self.plot = None
        
        return super().__init__(**kwargs)

    def create_factor_list(self,
                           column_name):

        factor_list = pd.unique(self.df[column_name]).tolist()
        return (factor_list)

    def create_color_mapper(self,
                            factors_to_map,
                            palette=Spectral6):

        color_mapper = CategoricalColorMapper(factors = factors_to_map, palette = palette)
        return (color_mapper)

    def set_source_1(self):

        self.source = ColumnDataSource(data = { 'x' : self.df['fertility'],
                                               'y' : self.df['life'],
                                               'country' : self.df['Country']})

    def set_source_2(self):

        self.source = ColumnDataSource(data = { 'x' : self.df.loc[1970].fertility,
                                                'y' : self.df.loc[1970].life,
                                                'country' : self.df.loc[1970].Country,
                                                'pop' : (self.df.loc[1970].population / 20000000) + 2,
                                                'region' : self.df.loc[1970].region,
                                                })

    def set_min_max_x(self):

        xmin,xmax = min(self.df.fertility), max(self.df.fertility)
        return (xmin, xmax)

    def set_min_max_y(self):

        ymin, ymax = min(self.df.life), max(self.df.life)
        return (ymin, ymax)

    def create_figure_2(self,
                        title,
                        x_range,
                        y_range,
                        x_axis_label,
                        y_axis_label,
                        color_factor,
                        color_mapper,
                        plot_height = 400,
                        plot_width = 700):

        plot = figure(title = title, 
                      plot_height = plot_height, 
                      plot_width = plot_width, 
                      x_range=x_range, 
                      y_range = y_range)
        plot.circle(x = 'x', 
                    y = 'y', 
                    fill_alpha = 0.8, 
                    source = self.source,
                    color = dict(field = color_factor, transform = color_mapper), 
                    legend = color_factor)
        plot.xaxis.axis_label = x_axis_label
        plot.yaxis.axis_label = y_axis_label
        plot.legend.location = 'top_right'

        return(plot)

    def create_slider(self,
                      slider_start_val,
                      slider_end_val,
                      slider_step_val,
                      slider_value,
                      slider_title):

        slider = Slider(start = slider_start_val,
                       end = slider_end_val,
                       step = slider_step_val,
                       value = slider_value,
                       title = slider_title)
        slider.on_change('value', self.callback_on_change_slider)

        return(slider)

    def create_hovertool(self,
                         label,
                         source_column): # recall, source column begins with '@' e.g. '@Country'

        hover = HoverTool(tooltips = [(label, source_column)])
        return (hover)

    def create_select(self,
                      options_list,
                      value,
                      title):

        select = Select(options = options_list, value = value, title = title)
        select.on_change('value', self.callback_on_change_slider)
        return (select)

    def set_x_select(self,
                     select):

        self.x_select = select

    def set_y_select(self,
                     select):

        self.y_select = select

    def set_slider(self,
                   slider):

        self.slider = slider

    def set_plot(self,
                 plot):
        
        self.plot = plot

    def create_figure_1(self):

        p = figure(title = '1970', x_axis_label = 'Fertility (children per woman)',
                   y_axis_label='Life Expectancy (years)',
                   plot_height=400, plot_width = 700,
                   tools = [HoverTool(tooltips = '@country')])
        p.circle(x = 'x', y = 'y', source = self.source)

        return (p)

    def output_file_show(self,
                         output_html_filename,
                         p):

        output_file(output_html_filename)
        show(p)

    def bokeh_serve_document(self,
                             plot,
                             title = 'Bokeh Served Document'):

        curdoc().add_root(plot)
        curdoc().title = 'Gapminder'

    def callback_on_change_slider(self,
                                  attr,
                                  old,
                                  new):

        yr = self.slider.value
        x = self.x_select.value
        y = self.y_select.value
        self.plot.xaxis.axis_label = x
        self.plot.yaxis.axis_label = y
        new_data = {
            'x' : self.df.loc[yr][x],
            'y' : self.df.loc[yr][y],
            'country' : self.df.loc[yr].Country,
            'pop' : (self.df.loc[yr].population / 20000000) + 2,
            'region' : self.df.loc[yr].region,
            }
        self.source.data = new_data
        self.plot.x_range.start = min(self.df[x])
        self.plot.x_range.end = max(self.df[x])
        self.plot.y_range.start = min(self.df[y])
        self.plot.y_range.end = max(self.df[y])

        self.plot.title.text = 'Gapminder data for %d' % yr

bokeh_ga = BokehGapminderApp()
bokeh_ga.set_source_2()
x_range = bokeh_ga.set_min_max_x()
y_range = bokeh_ga.set_min_max_y()
factor_list = bokeh_ga.create_factor_list('region')
color_mapper = bokeh_ga.create_color_mapper(factor_list, 
                                            palette=Spectral6)
p = bokeh_ga.create_figure_2(title = 'Gapminder Data for 1970',
                             x_range = x_range,
                             y_range = y_range,
                             color_factor = 'region',
                             color_mapper=color_mapper,
                             x_axis_label = 'Fertility (children per woman)',
                             y_axis_label = 'Life Expectancy (years)')
slider_obj = bokeh_ga.create_slider(1970, 2010, 1, 1970, 'Year')
x_select_obj = bokeh_ga.create_select(options_list = ['fertility', 'life', 'child_mortality', 'gdp'],
                                      value = 'fertility',
                                      title = 'x-axis data')

y_select_obj = bokeh_ga.create_select(options_list = ['fertility', 'life', 'child_mortality', 'gdp'],
                                      value = 'life',
                                      title = 'y-axis data')
hovertool_obj = bokeh_ga.create_hovertool('Country', '@country')
p.add_tools(hovertool_obj)
bokeh_ga.set_slider(slider_obj)
bokeh_ga.set_x_select(x_select_obj)
bokeh_ga.set_y_select(y_select_obj)
bokeh_ga.set_plot(p)
layout = row(widgetbox(slider_obj, x_select_obj, y_select_obj), p)
bokeh_ga.bokeh_serve_document(layout,
                              title = 'Gapminder')

