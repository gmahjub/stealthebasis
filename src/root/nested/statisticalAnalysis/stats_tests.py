from scipy import stats
import numpy as np
from root.nested import get_logger
import matplotlib.pyplot as plt


class StatsTests(object):

    LOGGER = get_logger()

    def __init__(self):

        StatsTests.LOGGER.info("StatsTests.__init__(): running function...")

    @staticmethod
    def ks_test_example(mean_of_norm_dist=0.0,
                        std_norm_dist=1.0,
                        size_dist=10000):

        x = np.random.normal(loc=mean_of_norm_dist, scale=std_norm_dist, size=size_dist)
        print (np.mean(x), np.std(x))
        plt.hist(x, bins=int(np.ceil(np.sqrt(size_dist))))
        ks_test_stat_np, p_value_np = stats.kstest(rvs = x, cdf='norm',
                                                   args=(mean_of_norm_dist, std_norm_dist), N = size_dist)
        plt.title("np.random.normal \n ks_test_stat = " + str(ks_test_stat_np) + "\n p_value = " + str(p_value_np))
        plt.show()
        z = stats.norm.rvs(loc=mean_of_norm_dist, scale=std_norm_dist, size=size_dist)
        print(np.mean(z), np.std(z))
        ks_test_stat_sp, p_value_sp = stats.kstest(rvs=z, cdf='norm',
                                                   args=(mean_of_norm_dist, std_norm_dist), N=size_dist)
        plt.hist(z, bins=int(np.ceil(np.sqrt(size_dist))))
        plt.title("Stats.norm.rvs \n ks_test_stat = " + str(ks_test_stat_sp) + "\n p_value = " + str(p_value_sp))
        plt.show()
        # the KS Test Statistics is the difference in the Cumulative Relative
        # frequency between observed distribution and the theoretical distribution.
        # IE. the RVS distribution, and the theoretical CDF (params in caps).

    @staticmethod
    def ks_test(rvs,
                dist_size,
                cdf='norm'):

        mu = np.mean(rvs)
        sigma = np.std(rvs)
        theo_rvs = stats.norm.rvs(loc = mu, scale = sigma, size = dist_size)
        StatsTests.LOGGER.info("StatsTest.ks_test(): RVS mean is %s, RVS StdDev is %s ", str(mu), str(sigma))
        args = (mu, sigma)
        ks_test_stat, p_value = stats.kstest(rvs=rvs, cdf='norm', args=args, N=dist_size)
        return ks_test_stat, p_value

    @staticmethod
    def shapiro_test(returns):
        # using shapiro test, check to see if the returns distribution is normal
        shapiro_test_stat, p_value = stats.shapiro(returns)
        return shapiro_test_stat, p_value


if __name__ == '__main__':

    StatsTests.ks_test_example()
    # PPF = percent probability function = 1 - cdf
    print(stats.norm.ppf(q=0.95, loc = 0, scale = 1))
    print(stats.norm.cdf(stats.norm.ppf(q=0.95, loc=0, scale=1)))
    print ("The first print statement gives us the 95th percentile value."
           "Which means that 5% of possible values will be that value or"
           "greater.")
    print ("The second print statement gives us the percent of values that will"
           "be at most that value. So if first print statement return 1.64485,"
           "then second print statement returns 95%.")
    # The PDF (Probability Density Function) should outline the histogram.
    # To produce the PDF, the y-axis should be normalized probabilities,
    # and the x-axis should be the values from the distribution.
    # IE. the x-axis should be output (across all values of q-param) of PPF func,
    # and the y-axis should be the output of CDF function at x[i] minus output of
    # CDF function at x[i-1]. This is done in the PDF function, but we can verify.
    x = np.linspace(stats.norm.ppf(0.01, loc = 0, scale = 1),
                    stats.norm.ppf(0.99, loc = 0, scale = 1),
                    100)
    fig, ax = plt.subplots(1,1)
    ax.plot(x, stats.norm.pdf(x))
    plt.show()
    stats.norm.pdf()
