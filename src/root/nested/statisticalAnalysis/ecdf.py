import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns

from root.nested import get_logger

class ECDF(object):
    """description of class"""

    # Emperical Cumulative Distribution Function

    def __init__(self, **kwargs):

        self.logger = get_logger()
        self.percentiles = None

        self.logger.info("ECDF.__init__.kwargs: %s", str(kwargs))

        for key,value in kwargs.items():

            if key == 'data':
                self.data = value
                self.logger.info("ECDF.__init__.data: %s", str(type(self.data))) # numpy array
            elif key == 'percentiles':
                self.percentiles = np.array(value)
                self.logger.info("ECDF.__init__.percentiles: %s", str(self.percentiles)) # a list
    
        n = len(self.data)
        x = np.sort(self.data)
        y = np.arange(1, n+1) / n

        if (self.percentiles is None):
            self.percentiles = np.array([2.5,25,50,75,97.5])
        self.data_ptiles = np.percentile(self.data, self.percentiles)

        self.x_data = x
        self.y_data = y

        #return super().__init__(**kwargs)

    def plot_ecdf(self,
                  xlabel,
                  ylabel,
                  legend_tuple):

        plt.plot(self.x_data, self.y_data, marker = '.', linestyle = 'none' )
        _ = plt.xlabel(xlabel)
        _ = plt.ylabel(ylabel)
        _ = plt.legend(legend_tuple, loc = 'lower right')
        _ = plt.plot(self.data_ptiles, self.percentiles/100, marker = 'D', color = 'red', linestyle = 'none')
        plt.show()

    def get_x_data(self):

        return (self.x_data)

    def get_y_data(self):

        return (self.y_data)
 
if __name__ == '__main__':

    ecdf = ECDF()
    

