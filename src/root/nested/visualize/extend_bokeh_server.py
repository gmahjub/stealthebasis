from bokeh.io import curdoc
from bokeh.layouts import widgetbox
from bokeh.models import Slider
from bokeh.plotting import figure

class extend_bokeh_server(object):
    """description of class"""

    def __init__(self, **kwargs):
        return super().__init__(**kwargs)

    def simple_test(self):

        plot = figure()
        plot.line([1,2,3,4,5], [2,5,4,6,7])
        curdoc().add_root(plot)


    # create plots and widgets 


    # add callbacks


    # arrange plots and widgets in layouts

    curdoc().add_root(layout)



