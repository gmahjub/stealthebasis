import numpy as np
import matplotlib.pyplot as plt
from root.nested import get_logger


class ECDF(object):
    """description of class"""

    # Emperical Cumulative Distribution Function

    def __init__(self, **kwargs):

        self.logger = get_logger()
        self.percentiles = None
        self.title = None

        for key,value in kwargs.items():

            if key == 'data':
                self.data = value
                # self.logger.info("ECDF.__init__.data: %s", str(type(self.data))) # numpy array
            elif key == 'percentiles':
                self.percentiles = np.array(value)
                self.logger.info("ECDF.__init__.percentiles: %s", str(self.percentiles)) # a list
            elif key == 'title':
                self.title = title
                self.logger.info("ECDF.__init__.title: %s", str(self.title))
    
        n = len(self.data)
        x = np.sort(self.data)
        y = np.arange(1, n+1) / n
        self.mu = np.mean(self.data)
        self.sigma = np.std(self.data)

        if self.percentiles is None:
            self.percentiles = np.array([2.5,25,50,75,97.5])
        self.data_ptiles = np.percentile(self.data, self.percentiles)

        self.x_data = x
        self.y_data = y

    def plot_ecdf(self,
                  xlabel,
                  ylabel,
                  title='ECDF',
                  legend_tuple=None):

        if self.title is None:
            self.title = title
        elif self.title is not None and title is not 'ECDF':
            self.title = title
        plt.plot(self.x_data, self.y_data, marker = '.', linestyle = 'none' )
        _ = plt.xlabel(xlabel)
        _ = plt.ylabel(ylabel)
        _ = plt.title(self.title)
        if legend_tuple is not None:
            _ = plt.legend(legend_tuple, loc = 'lower right')
        _ = plt.plot(self.data_ptiles, self.percentiles/100, marker = 'D', color = 'red', linestyle = 'none')
        plt.show()

    def get_x_data(self):

        return self.x_data

    def get_y_data(self):

        return self.y_data

    def get_data_ptiles(self):

        return self.data_ptiles

    def get_mu(self):

        return self.mu

    def get_sigma(self):

        return self.sigma


if __name__ == '__main__':

    ecdf = ECDF()
