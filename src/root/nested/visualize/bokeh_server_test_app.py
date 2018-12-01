from random import random

from bokeh.layouts import column, widgetbox, row
from bokeh.models import Button, Slider, Select
from bokeh.palettes import RdYlBu3
from bokeh.plotting import figure, curdoc, ColumnDataSource

import numpy as np
import pandas as pd
from numpy.random import random

def create_bokeh_test_app_2():

    plot = figure()
    plot.line([1,2,3,4,5], [2,5,4,6,7])
    return (plot)

def create_bokeh_test_app_3():

    slider = Slider(title = 'my slider', start = 0, end = 10, step = 0.1, value = 2)
    layout = widgetbox(slider)
    return (layout)

def create_bokeh_test_app_4():

    slider1 = Slider(title = 'slider1', start = 0, end = 10, step = 0.1, value = 2)
    slider2 = Slider(title = 'slider2', start = 10, end = 100, step = 1, value = 20)
    layout = widgetbox(slider1, slider2)
    return (layout)

def callback_bokeh_test_app_5(attr, old, new):

    scale = new
    new_y = np.sin(scale/source.data['x'])
    source.data['y'] = new_y

def create_bokeh_test_app_5():

    global source

    x = np.linspace(0.3, 10, 300)
    y = np.sin(x)
    source = ColumnDataSource(data = {'x':x, 'y': y})
    plot = figure()
    plot.line('x', 'y', source = source)
    slider = Slider(title = 'my slider', start = 0, end = 10, step = 0.1, value = 2)
    slider.on_change('value', callback_bokeh_test_app_5)
    layout = column(widgetbox(slider), plot)
    return (layout)

def callback_bokeh_test_app_6(attr, old, new):

    if new == 'female_literacy':
        source.data = { 'x' : fertility,
                       'y' : female_literacy 
                       }
    else:
        source.data = {
            'x' : fertility,
            'y' : population}

def create_bokeh_test_app_6():

    global source, fertility, female_literacy, population

    datacamp_data_path = 'C:\\Users\\ghazy\\workspace\\data\\datacamp\\'
    df = pd.read_csv(datacamp_data_path + 'literacy_birth_rate.csv', sep = ',', header = 0)
    fertility = df['fertility']
    female_literacy = df['female literacy']
    population = df['population']
    source = ColumnDataSource(data = {'x' : fertility,
                                      'y' : female_literacy})
    plot = figure()
    plot.circle('x', 'y', source = source)
    select = Select(title = 'distribution', options = ['female_literacy', 'population'], value = 'female_literacy')
    select.on_change('value', callback_bokeh_test_app_6)
    layout = row(select, plot)

    return (layout)

def callback_bokeh_test_app_7(attr, old, new):

    if select1.value == 'A':
        select2.options = ['1', '2', '3']
        select2.value = '1'
    else:
        select2.options = ['100', '200', '300']
        select2.value = '100'

def create_bokeh_test_app7():

    select1 = Select(title = 'First', options = ['A', 'B'], value = 'A')
    select2 = Select(title = 'Second', options = ['1', '2', '3'], value = '1')

# create a callback that will add a number in a random location
def callback_bokeh_test_app_1():
    global i

    # BEST PRACTICE --- update .data in one step with a new dict
    new_data = dict()
    new_data['x'] = ds.data['x'] + [random()*70 + 15]
    new_data['y'] = ds.data['y'] + [random()*70 + 15]
    new_data['text_color'] = ds.data['text_color'] + [RdYlBu3[i%3]]
    new_data['text'] = ds.data['text'] + [str(i)]
    ds.data = new_data

    i = i + 1

def create_bokeh_test_app_1():
    
    global i
    global ds

    # create a plot and style its properties
    p = figure(x_range=(0, 100), y_range=(0, 100), toolbar_location=None)
    p.border_fill_color = 'black'
    p.background_fill_color = 'black'
    p.outline_line_color = None
    p.grid.grid_line_color = None

    # add a text renderer to our plot (no data yet)
    r = p.text(x=[], y=[], text=[], text_color=[], text_font_size="20pt",
           text_baseline="middle", text_align="center")
    i = 0
    ds = r.data_source

    return (p)

def run_bokeh_test_app_1():

    p = create_bokeh_test_app_1()
    # add a button widget and configure with the call back
    button = Button(label="Press Me")
    button.on_click(callback_bokeh_test_app_1)
    # put the button and plot in a layout and add to the document
    curdoc().add_root(column(button, p))

def run_bokeh_test_app_2():
    plot = create_bokeh_test_app_2()
    curdoc().add_root(plot)

def run_bokeh_test_app_3():
    layout = create_bokeh_test_app_3()
    curdoc().add_root(layout)

def run_bokeh_test_app_4():
    layout = create_bokeh_test_app_4()
    curdoc().add_root(layout)

def run_bokeh_test_app_5():
    layout = create_bokeh_test_app_5()
    curdoc().add_root(layout)

def run_bokeh_test_app_6():
    layout = create_bokeh_test_app_6()
    curdoc().add_root(layout)

#run_bokeh_test_app_6()
def run_bokeh_test_app_7():
    select1 = Select(title = 'First', options = ['A', 'B'], value = 'A')
    select2 = Select(title = 'Second', options = ['1', '2', '3'], value = '1')

    select1.on_change('value', callback_bokeh_test_app_7)
    layout = widgetbox(select1, select2)
    curdoc().add_root(layout)

run_bokeh_test_app_6()