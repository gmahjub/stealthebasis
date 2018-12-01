import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class ExtendSeaborn(object):
    """description of class"""

    def __init__(self, **kwargs):
        
        sns.set()
        return super().__init__(**kwargs)
    
    def sns_regplot_impl(self,
                        x_data, # column name from df (str)
                        y_data, # column name from df (str)
                        df,
                        order = 1, # 1st order is default, 1st order = linear regression = lmplot()
                        plot_label = 'plot_label',
                        incl_scatter_plot = True,
                        line_color = 'red'):

        sns.regplot(x = x_data, y = y_data, data = df, scatter=incl_scatter_plot, color = line_color, label = plot_label, order = order)
        plt.legend(loc = 'upper right')
        plt.show()

    def sns_residplot_impl(self,
                           x_data, # column name from df (str)
                           y_data, # column name from df (str)
                           df,
                           scatter_color = 'green'):

        sns.residplot(x = x_data, y = y_data, data = df, color = scatter_color)
        plt.show()

    def sns_single_multi_lmplot_impl(self,
                                     x_data,
                                     y_data,
                                     df,
                                     group_by_field,
                                     palette = 'Set1'):

        sns.lmplot(x = x_data, y = y_data, data = df, hue = group_by_field, palette = palette)
        plt.show()

    def sns_multi_col_lmplot_impl(self,
                                  x_data,
                                  y_data,
                                  df,
                                  group_by_field):

        sns.lmplot(x = x_data, y = y_data, data = df, col = group_by_field)
        plt.show()

    def sns_multi_row_lmplot_impl(self,
                                  x_data,
                                  y_data,
                                  df,
                                  group_by_field):

        sns.lmplot(x = x_data, y = y_data, data = df, row = group_by_field)
        plt.show()

    def sns_lmplot_impl(self,
                        x_data, # column name from df (str)
                        y_data, # column name from df (str)
                        df):

        sns.lmplot(x = x_data, y = y_data, data = df)
        plt.show()

    def sns_pairplot_impl(self,
                          df,
                          group_by_col,
                          kind = 'reg'):

        sns.pairplot(df, kind = kind, hue = group_by_col) # default for kind is "scatter" in sns.pairplot
        plt.show()

    def sns_hist_plt(self,
                     np_array,
                     x_label=None,
                     y_label=None):

        self.square_root_rule_bins(len(np_array))
        plt.hist(np_array)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()

    def square_root_rule_bins(self,
                              len_data):

        return (np.sqrt(len_data))

    def sns_swarmplot_impl(self,
                           df, # pandas dataframe object
                           x_data, # string column name from dataframe df
                           y_data): # string column name from dataframe df

        sns.swarmplot(x = x_data, y = y_data, data = df)
        # using label names same as dataframe column names
        _ = plt.xlabel(x_data)
        _ - plt.ylabel(y_data)
        plt.show()