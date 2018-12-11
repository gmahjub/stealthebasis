import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from root.nested.statisticalAnalysis.ecdf import ECDF
from root.nested import get_logger


class HackerStats(object):
    """description of class"""

    def __init__(self, **kwargs):

        self.logger = get_logger()
        return super().__init__(**kwargs)

    def seed_rand_num_generator(self):

        np.random.seed(42)

    def gen_array_rand_nums(self,
                            size=1):

        # seed the random number generator first
        self.seed_rand_num_generator()
        return (np.random.random(size = size))

    def perform_bernoulli_trials(self, n, p):

        n_success = 0
        for i in range(n):
            random_number = np.random.random()
            if random_number < p:
                n_success += 1
        
        return (n_success)

    def sample_binomial_dist(self, n, p, size = 10000):

        # this function combines perform_bernoulli_trials and the for loop over 1000
        # in the main funciton below
        n_defaults = np.random.binomial(n, p, size = size)
        self.plot_simulation_ecdf(n_defaults,
                                  x_label = "num_defaults/100",
                                  y_label = "CDF (%/100) (BinomialDist)")

    def sample_poisson_dist(self, np, size = 10000):

        # remember, Poisson is a specific case of binoomial, where p is small and n is large
        # Poisson is a limit of the Binomial distribution, for rare events.
        samples_poisson = np.random.poisson(np, size = size)
        self.plot_simulation_ecdf(samples_poisson,
                                  x_label = "num_defaults/100",
                                  y_label = "CDF (%/100) (PoissonDist)")

    def sample_exponential_dist(self, mu_param, size = 100000):

        # exponential distribution simulates the time between poisson distributed events,
        # i.e. the time between rare events can be simulation via the exponential distribution
        samples_exp = np.random.exponential(scale = mu_param, size = size)
        self.plot_simulation_ecdf(samples_exp,
                                  x_label = "x",
                                  y_label = "y (Exp Dist")

    def sample_normal_dist(self, mu, sigma, size = 100000):

        samples = np.random.normal(loc = mu, scale = sigma, size = size)
        self.plot_simulation_ecdf(samples,
                                  x_label = 'x',
                                  y_label = 'y (NormalDist)')

    def sample_log_normal_dist(self, mu, sigma, size = 100000):

        samples = np.random.lognormal(mean=mu, sigma=sigma, size=size)
        self.plot_simulation_ecdf(samples,
                                  x_label='x',
                                  y_label='y (LogNormalDist)')

    def sample_gamma_dist(self, k, theta, size=100000):

        samples = np.random.gamma(shape=k, scale=theta, size=size)
        self.plot_simulation_ecdf(samples,
                                  x_label='x',
                                  y_label='y (GammaDist)')

    def sample_weibull_dist(self, lam, k, size=100000):

        samples = np.random.weibull(a=k, size=size)*lam
        self.plot_simulation_ecdf(samples,
                                  x_label='x',
                                  y_label='y (WeibullDist)')

    def get_num_bins_hist(self,
                          sample_size):

        return np.sqrt(sample_size)

    def check_normality(self,
                        data):

        # check normality graphically

        mu = np.mean(data)
        sigma = np.std(data)
        samples = np.random.normal(loc=mu, scale=sigma, size=10000)
        ecdf_theor = ECDF(data=samples)
        ecdf_data = ECDF(data=data)

        _ = plt.plot(ecdf_theor.x_data, ecdf_theor.y_data)
        _ = plt.plot(ecdf_data.x_data, ecdf_data.y_data, marker = '.', linestyle = 'none')
        _ = plt.xlabel('x')
        _ = plt.ylabel('y)')
        plt.show()

    def plot_simulation_ecdf(self,
                             data,
                             x_label,
                             y_label,
                             title='ECDF'):

        self.logger.info("HackertStats.plot_simulation_ecdf.data: %s ", str(data))
        ecdf = ECDF(data = data)
        ecdf.plot_ecdf(x_label, y_label, title)
        return (ecdf)

    def plot_pdf(self,
                 data,
                 x_label = 'x',
                 ylabel = 'PDF',
                 bins = 50):

        _ = plt.hist(data, histtype = 'step', normed = True, bins = 50)
        _ = plt.xlabel(x_label)
        _ = plt.ylabel(y_label)
        plt.show()

    def linear_regression(self,
                          x_data,
                          y_data,
                          x_label,
                          y_label):

        _ = plt.plot(x_data, y_data, marker = ".", linestyle = 'none')
        plt.margins(0.02)
        _ = plt.xlabel(xlabel)
        _ = plt.ylabel(ylabel)
        a,b = np.polyfit(x_data, y_data, 1)
        return(a,b)

    def bootstap_linregress_params(self,
                                   x_data,
                                   y_data,
                                   size = 1000,
                                   conf_int = [2.5, 97.5]):

        """This function is just like the parameter estimation function for beak depth below.
        Here, we are estimating the linear regression parameters (slope, intercept) for the
        comparion of beak length vs. beak depth. We will estimate the paramters using
        np.polyfit for 1975 and 2012 data. Then we will boostrap in order to come up with
        confidence intervals.
        """
        slope, intercept = np.polyfit(x_data, y_data, 1) # first order polynomical regression = linear
        bs_slope_reps, bs_intercept_reps = self.draw_bootstrap_pairs_linregress(x_data,
                                                                                y_data,
                                                                                size = size)
        bs_conf_int_slope = np.percentile(bs_slope_reps, conf_int)
        bs_conf_int_intercept = np.percentile(bs_intercept_reps, conf_int)
        #_ = plt.hist(bs_slope_reps, bins = 50, normed = True)
        #_ = plt.xlabel('slope')
        #_ = plt.ylabel('PDF')
        #plt.show()
        return(slope, intercept, bs_conf_int_slope, bs_conf_int_intercept, bs_slope_reps, bs_intercept_reps)

    def plot_bootstrap_linregress(self,
                                  bs_slope_reps_list,
                                  bs_intercept_reps_list,
                                  x_data_list,
                                  y_data_list,
                                  color_list,
                                  legend_label_list,
                                  x_label,
                                  y_label,
                                  how_many_lines=100, # must be less than then total number of boostrap replicates
                                  bs_line_x_vals = np.array([0,100])):

        """Plot scatter plot of data, and then also plot linear regression bootstrap. Why?
           This plots the possible linear regresesions given by the bootstrap, the potential
           values for the linear regression parameters, specfically slope and intercept
        """
        xy_data_list = zip(x_data_list, y_data_list, color_list)
        resultSet = list(xy_data_list)
        for xy_tuple in resultSet:
            x_data = xy_tuple[0]
            y_data = xy_tuple[1]
            color = xy_tuple[2]
            _ = plt.plot(x_data, y_data, marker = '.', linestyle = 'none', color = color, alpha = 0.5 )
        _ = plt.xlabel(x_label)
        _ = plt.ylabel(y_label)
        _ = plt.legend(legend_label_list, loc = 'upper left')
        x = np.array(bs_line_x_vals)

        bs_data_list = zip(bs_slope_reps_list, bs_intercept_reps_list, color_list)
        resultSet = list(bs_data_list)
        for rs in resultSet:
            bs_slope_reps = rs[0]
            bs_intercept_reps = rs[1]
            color = rs[2]
            for i in range(how_many_lines):
                _ = plt.plot(x, bs_slope_reps[i]*x + bs_intercept_reps[i], linewidth = 0.5, alpha = 0.2, color = color)

        plt.margins(0.02)
        plt.show()

    def draw_bootstrap_reps(self,
                            data,
                            func,
                            size = 1):

        """This function is used by other functions in this class to create
           bootstrap replicates of the input data. It simply pulls random entries
           from the input data, with replacement (meaning each entry in input dat
           can appear in output bootstrap replicate more than once. The function used
           to do this randomized pulling is np.random.choice().
        """
        self.seed_rand_num_generator()
        bs_replicates = np.empty(shape = size)
        for i in range(size):
            bs_replicates[i] = func(np.random.choice(data, size = len(data)))
        return (bs_replicates)

    def draw_bootstrap_pairs_linregress(self,
                                        x_data,
                                        y_data,
                                        size = 1):

        inds = np.arange(len(x_data))
        bs_slope_reps = np.empty(size)
        bs_intercept_reps = np.empty(size)
        for i in range(size):
            bs_inds = np.random.choice(inds, size = len(inds))
            bs_x, bs_y = x_data[bs_inds], y_data[bs_inds]
            bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x, bs_y, 1)
        return (bs_slope_reps, bs_intercept_reps)

    def prepare_finch_beak_case_study_data(self):

        """Specific function to prepare the finch beak case study data. Simply read
           in the data from csv into Panda's dataframe, and then return individual columns
           of data as numpy arrays.
        """

        finch_beak_1975 = "C:\\Users\\ghazy\\workspace\\data\\datacamp\\finch_beaks_1975.csv"
        finch_beak_2012 = "C:\\Users\\ghazy\\workspace\\data\\datacamp\\finch_beaks_2012.csv"
        df_1975 = pd.read_csv(finch_beak_1975, sep = ',', header = 0)
        df_2012 = pd.read_csv(finch_beak_2012, sep = ',', header = 0)
        bd_1975 = df_1975.bdepth.values
        bl_1975 = df_1975.blength.values
        bd_2012 = df_2012['Beak depth, mm'].values
        bl_2012 = df_2012['Beak length, mm'].values

        return (bd_1975, bd_2012, bl_1975, bl_2012)

    def prepare_heritability_case_study_data(self):

        scandens_csv = "C:\\Users\\ghazy\\workspace\\data\\datacamp\\scandens_beak_depth_heredity.csv"
        fortis_csv = "C:\\Users\\ghazy\\workspace\\data\\datacamp\\fortis_beak_depth_heredity.csv"
        df_scandens = pd.read_csv(scandens_csv, sep = ",", header = 0)
        df_fortis = pd.read_csv(fortis_csv, sep = ",", header = 0)
        bd_offspring_scandens = df_scandens.mid_offspring.values
        bd_parent_scandens = df_scandens['mid_parent'].values
        bd_offspring_fortis = df_fortis['Mid-offspr'].values
        bd_parent_fortis = (df_fortis['Male BD'].values + df_fortis['Female BD'].values)/2.0
        return (bd_offspring_scandens, bd_parent_scandens, bd_offspring_fortis, bd_parent_fortis)

    def eda_heritability(self,
                         bd_offspring_scandens,
                         bd_parent_scandens,
                         bd_offspring_fortis,
                         bd_parent_fortis):

        _ = plt.plot(bd_parent_fortis, bd_offspring_fortis, marker = '.', linestyle = 'none', color = 'blue', alpha = 0.5)
        _ = plt.plot(bd_parent_scandens, bd_offspring_scandens, marker = '.', linestyle = 'none', color = 'red', alpha = 0.5)
        _ = plt.xlabel('parental beak depth (mm)')
        _ = plt.ylabel('offspring beak depth (mm)')
        _ = plt.legend(['G. fortis', 'G. Scandens'], loc = 'lower right')
        plt.show()

    def plot_finch_beak_ecdf(self,
                             bd_1975,
                             bd_2012):

        """Using the ECDF class, plot the ECDF of the two sets of finch beak data,
           one from 1975 and the other from 2012.
        """

        bd_1975_ecdf = ECDF(data = bd_1975)
        bd_2012_ecdf = ECDF(data = bd_2012)
        line_1975, = plt.plot(bd_1975_ecdf.get_x_data(), bd_1975_ecdf.get_y_data(), marker = '.', linestyle = 'none')
        line_2012, = plt.plot(bd_2012_ecdf.get_x_data(), bd_2012_ecdf.get_y_data(), marker = '.', linestyle = 'none')
        plt.margins(0.02)
        _ = plt.xlabel('beak depth (mm)')
        _ = plt.ylabel('ECDF')
        _ = plt.legend([line_1975, line_2012], ['1975', '2012'], loc = 'lower right')
        plt.show()

    def param_estimate_mean_beak_depth(self,
                                       bd_1975,
                                       bd_2012):

        """ Here, we are estimating the difference in mean beak depth. That's right,
            the observed difference in means is just an estimate, since they are measurements taken
            and therefore are just a sample. In order to figure out the range of possible values for
            each of bd_1975 and bd_2012, we need to bootstrap. We get the conf int by bootstrapping,
            which then tells us how confident we can be that the actual values will be within a 
            certain range.
        """

        # calculate the actual difference in average beak depth
        mean_diff = np.mean(bd_2012) - np.mean(bd_1975)
        # now, bootstrap each of the data sets, to see if its possible for us to get a significantly different mean
        bs_replicates_1975 = self.draw_bootstrap_reps(bd_1975, np.mean, size = 10000)
        bs_replicates_2012 = self.draw_bootstrap_reps(bd_2012, np.mean, size = 10000)
        # for each replicate of the 10,000, calcuate the difference between 2012 and 1975
        bs_diff_replicates = bs_replicates_2012 - bs_replicates_1975
        # calcualte a confidence interval to see the range of values that cover 95% of the possibilities
        conf_int = np.percentile(bs_diff_replicates, [2.5, 97.5])
        print('difference of means = ', mean_diff, 'mm')
        print('95% confidence interval = ', conf_int, 'mm')

    def hypothesis_test_mean_beak_depth(self,
                                        bd_1975,
                                        bd_2012):

        """ Here, we want to test a hypothesis, that beaks in 2012 are larger on average than
            beaks in 1975. We do this by setting the mean of both datasets to be equal (de-meaning
            each sample and then adding the combined mean to each sample). Then, we bootstrap that
            data, to see how often would we would get the observed (sampled) difference in mean (mean_diff)
            if the means were the same? So we purposely set the means to be the same, and then see how
            often we would get mean_diff or greater during bootstrapping. If p value is high, then its possible
            that the mean_diff from our sample could have happened even if the means were not different.
            That would therefore nullify our hypothesis that beak depth is greater in 2012 than 1975.
        """

        mean_diff = np.mean(bd_2012) - np.mean(bd_1975)
        print ("mean diff =", mean_diff)
        # first step is to make both arrays (bd_1975, bd_2012) have the same mean,
        # so calculate the combined mean
        combined_mean = np.mean(np.concatenate((bd_1975, bd_2012)))
        # next, de-mean and add new mean
        bd_1975_shifted = bd_1975 - np.mean(bd_1975) + combined_mean
        bd_2012_shifted = bd_2012 - np.mean(bd_2012) + combined_mean
        # next, bootstrap
        bs_replicates_1975 = self.draw_bootstrap_reps(bd_1975_shifted, np.mean, size = 10000)
        bs_replicates_2012 = self.draw_bootstrap_reps(bd_2012_shifted, np.mean, size = 10000)
        # compute the difference of the means of the boostrap replicates
        bs_diff_replicates = bs_replicates_2012 - bs_replicates_1975
        # calculate the p-value - the p value 
        p = np.sum(bs_diff_replicates >= mean_diff) / len(bs_diff_replicates)

        print ("p = ", p)

        return p

    def pairs_data_eda(self,
                       x1_data,
                       y1_data,
                       x2_data,
                       y2_data,
                       x_label,
                       y_label,
                       legend_label_list):

        """ Simple function to plot relatinship between two variables in a scatter.
        """
        _ = plt.plot(x1_data, y1_data, marker = '.', linestyle = 'none', color = 'blue', alpha = 0.5)
        _ = plt.plot(x2_data, y2_data, marker = '.', linestyle = 'none', color = 'red', alpha = 0.5)
        _ = plt.xlabel(x_label)
        _ = plt.ylabel(y_label)
        _ = plt.legend(legend_label_list, loc = 'upper left')
        plt.show()
    
    def run_pairs_bootstrap_example(self,
                                    x_data,
                                    y_data):

        return (1)

    def estimate_pearson_correlation(self,
                                     x_data_list,
                                     y_data_list,
                                     data_label_list,
                                     size = 1000,
                                     conf_int = [2.5, 97.5]):

        # pearson correlation using np.corrcoef()
        xy_data_list = zip(x_data_list, y_data_list, data_label_list)
        resultSet = list(xy_data_list)
        return_list = []
        for rs in resultSet:
            pearson_corr = np.corrcoef(rs[0], rs[1])[0,1]
            bs_replicate = self.draw_bootstrap_pairs_generic_func(rs[0], rs[1], pearson_r, size = 1000)
            bs_conf_int = np.percentile(bs_replicate, conf_int)
            return_tuple = (rs[2], pearson_r, bs_replicate, bs_conf_int)
            return_list.append(return_tuple)
            print (rs[2], ': correl :', pearson_corr, ' Confidence Interval', bs_conf_int)
        return (return_list)

    def draw_bootstrap_pairs_generic_func(self,
                                          x_data,
                                          y_data,
                                          func,
                                          size = 1):

        """Perform pairs bootstrap for a single statistic.
        """
        inds = np.arange(len(x_data))
        bs_replicates = np.empty(size)
        for i in range(size):
            bs_inds = np.random.choice(inds, size = len(inds))
            bs_x, bs_y = x_data[bs_inds], y_data[bs_inds]
            bs_replicates[i] = func(bs_x, bs_y)

        return bs_replicates

    def generate_permutation_sample(self,
                                    tuple_of_data_arrays): # tuple of np arrays, to be exact

        concated_data = np.concatenate(tuple_of_data_arrays)
        permuted_data = np.random.permutation(concated_data)
        list_of_permuted_arrays = []
        next_perm_start_idx =0
        for np_array in tuple_of_data_arrays:
            perm_sample = permuted_data[next_perm_start_idx:len(np_array)]
            list_of_permuted_arrays.append(perm_sample)
            next_perm_start_idx = len(np_array)
        return (list_of_permuted_arrays)


    def run_loan_default_example(self):

        success_prob = 0.05
        n_defaults = np.empty(1000)
        for i in range(1000):
            n_defaults[i] = hs.perform_bernoulli_trials(100, success_prob)

        _ = plt.hist(n_defaults, normed = True)
        _ = plt.xlabel('number of defaults out of 100 loans')
        _ = plt.ylabel('probability')
        plt.show()

        hs.plot_simulation_ecdf(data = n_defaults,
                                x_label = 'count',
                                y_label = '%/100')

        n_lose_money = np.sum(n_defaults >= 10)
        print ('Probability of losing money = ', n_lose_money / len(n_defaults))

def pearson_r(x,y):
        
        return (np.corrcoef(x,y)[0,1])


    # create a function that takes in stock returns data, and spits out a histogram
    # showing the likelihood of future returns by magnitute, from negative to positive.
    # this is Monte Carlo Simulation


if __name__ == '__main__':

    hs = HackerStats()
    #hs.run_loan_default_example()
    bd_1975, bd_2012, bl_1975, bl_2012 = hs.prepare_finch_beak_case_study_data()
    hs.plot_finch_beak_ecdf(bd_1975, bd_2012)
    hs.param_estimate_mean_beak_depth(bd_1975, bd_2012)
    hs.hypothesis_test_mean_beak_depth(bd_1975, bd_2012)
    hs.pairs_data_eda(bl_1975, bd_1975, bl_2012, bd_2012, 'beak length (mm)', 'beak depth (mm)', ['1975', '2012'])
    slope_1975, intercept_1975, slope_conf_1975, int_conf_1975, bs_slope_reps_1975, bs_intercept_reps_1975 = hs.bootstap_linregress_params(bl_1975, bd_1975, size = 1000)
    print('1975: slope = ', slope_1975, 'conf int =', slope_conf_1975)
    print('1975: intercept = ', intercept_1975, 'conf int =', int_conf_1975)
    slope_2012, intercept_2012, slope_conf_2012, int_conf_2012, bs_slope_reps_2012, bs_intercept_reps_2012 = hs.bootstap_linregress_params(bl_2012, bd_2012, size = 1000)
    print('2012: slope = ', slope_2012, 'conf int =', slope_conf_2012)
    print('2012: intercept = ', intercept_2012, 'conf int =', int_conf_2012)
    
    bs_slope_reps_list = [bs_slope_reps_1975, bs_slope_reps_2012]
    bs_intercept_reps_list = [bs_intercept_reps_1975, bs_intercept_reps_2012]
    x_data_list = [bl_1975, bl_2012]
    y_data_list = [bd_1975, bd_2012]
    color_list = ['blue', 'red']
    legend_label_list = ['1975', '2012']
    x_label = 'beak length (mm)'
    y_label = 'beak depth (mm)'
    bs_line_x_vals = np.array([7.5,17.5])

    hs.plot_bootstrap_linregress(bs_slope_reps_list,
                                 bs_intercept_reps_list,
                                 x_data_list,
                                 y_data_list,
                                 color_list,
                                 legend_label_list,
                                 x_label,
                                 y_label,
                                 bs_line_x_vals = bs_line_x_vals)

    bd_offspring_scandens, bd_parent_scandens, bd_offspring_fortis, bd_parent_fortis = hs.prepare_heritability_case_study_data()
    hs.eda_heritability(bd_offspring_scandens,
                        bd_parent_scandens,
                        bd_offspring_fortis,
                        bd_parent_fortis)
    x_data_list = [bd_parent_scandens, bd_parent_fortis]
    y_data_list = [bd_offspring_scandens, bd_offspring_fortis]
    data_label_list = ['G. Scandens', 'G. fortis']
    hs.estimate_pearson_correlation(x_data_list,
                                    y_data_list,
                                    data_label_list,
                                    size = 1000,
                                    conf_int = [2.5, 97.5])
