'''
Created on Dec 20, 2017

@author: ghazy
'''

from math import pow, sqrt
import numpy as np

from root.nested import get_logger
logger = get_logger()

class InterestRates(object):
    
    def __init__(self,
                 amountA,
                 compoundFreqPerAnnumM,
                 ratePerAnnumR,
                 numYearsInvestedN=1):
        
        self.amountA = amountA
        self.compoundFreqPerAnnumM = compoundFreqPerAnnumM
        self.ratePerAnnumR = ratePerAnnumR
        self.numYearsInvestedN = numYearsInvestedN
        
    def calc_terminal_value_of_investment(self):
        
        return self.amountA*pow((1 + (self.ratePerAnnumR/self.compoundFreqPerAnnumM)), self.compoundFreqPerAnnumM*self.numYearsInvestedN) 
    
    # continuous compounding yields very similar return to daily compounding
    def calc_continuous_compounding_terminal_value(self):
        
        return self.amountA*np.exp(self.ratePerAnnumR*self.numYearsInvestedN)
    
    # returns the equivalent continuously compounded rate
    def convert_to_continuous_compounding_rate(self,
                                               fromFreqM,
                                               fromRateR):
        
        return fromFreqM*np.log(1+(fromRateR/fromFreqM))
    
    def convert_to_non_continuous_compounding_rate(self,
                                                   toFreqM,
                                                   continuousRateR):
        
        return toFreqM*(np.exp(continuousRateR/toFreqM)-1)
    
    def determine_zero_treasury_rate(self):
        
        return 1
    
import scipy.optimize as optimize
    
class InterestRateInstrument(object):
    
    def __init__(self,
                 par_value,
                 term,
                 coupon,
                 price=1000.0,
                 compounding_freq=2,
                 zero_rate=0.0,
                 forward_rate=0.0,
                 obs_ytm=0.0):   # compounding_freq is per annum, default is 2 times a year, every 6 months
        
        self.par_value = par_value
        self.coupon = coupon
        self.price = price
        self.compounding_freq = compounding_freq
        self.term = term
        self.zero_rate = zero_rate
        self.forward_rate = forward_rate
        self.obs_ytm = obs_ytm
        self.calculated_ytm = 0.0
        
        self.p=print
        
    def set_par_value(self,
                      par_value):
        
        self.par_value = par_value
        
    def get_par_value(self):
        
        return self.par_value
    
    def set_term(self,
                 term):
        
        self.term = term
    
    def get_term(self):
        
        return self.term
    
    def set_coupon(self,
                   coupon):
        
        self.coupon = coupon
        
    def get_coupon(self):
        
        return self.coupon
    
    def set_price(self,
                  price):
        
        self.price = price
        
    def get_price(self):
        
        return self.price

    def set_compounding_freq(self,
                             compounding_freq):
        
        self.compounding_freq = compounding_freq
        
    def get_compounding_freq(self):
        
        return self.compounding_freq
    
    def get_zero_rate(self):
        
        return self.zero_rate
    
    def set_zero_rate(self,
                      zero_rate):
        
        self.zero_rate = zero_rate
        
    def set_forward_rate(self,
                         forward_rate):
        
        self.forward_rate = forward_rate
    
    def get_obs_ytm(self):
        
        return self.obs_ytm
    
    def set_obs_ytm(self,
                    obs_ytm):
        
        self.obs_ytm = obs_ytm
        
    def get_calculated_ytm(self):
        
        return self.calculated_ytm
    
    def set_calculated_ytm(self,
                           calculated_ytm):
        
        self.calculated_ytm = calculated_ytm
        
    def get_yield_to_maturity(self,
                              guess=0.05):
        
        freq = float(self.get_compounding_freq())
        par = self.get_par_value()
        price = self.get_price()
        periods = self.get_term()*freq
        coupon = self.get_coupon()
        coupon = coupon/100.*par/freq
        dt = [(i+1)/freq for i in range(int(periods))]
        ytm_func = lambda y: sum([coupon/(1+y/freq)**(freq*t) for t in dt]) + par/(1+y/freq)**(freq*self.get_term()) - price
        return optimize.newton(ytm_func, guess)
    
    def get_price_from_ytm(self,
                           ytm):
        
        if (float(self.get_coupon()) == 0.0):
            return self.get_price_from_ytm_zero_coupon(ytm)
        freq = float(self.get_compounding_freq())
        periods = self.get_term()*freq
        coupon = self.get_coupon()/100.*self.get_par_value()/freq
        par= self.get_par_value()
        dt = [(i+1)/freq for i in range(int(periods))]
        price = sum([coupon/(1+ytm/freq)**(freq*t) for t in dt]) + par/(1+ytm/freq)**(freq*self.get_term())    
        return price
    
    def get_price_from_ytm_zero_coupon(self,
                                       ytm):
        
        self.get_term()
    
    def get_iri_modified_duration(self,
                                  dy=0.01):
        
        ytm = self.get_yield_to_maturity()
        ytm_minus = ytm - dy
        price_minus = self.get_price_from_ytm(ytm_minus)
        ytm_plus = ytm + dy
        price_plus = self.get_price_from_ytm(ytm_plus)
        mduration = (price_minus-price_plus)/(2*self.get_price()*dy)
        return mduration
    
    #placeholder, come back to it
    # two bnonds that mature on the same date can have different convexity,
    # depending on where on the yield curve they lie.. e.g. a 30 year bond 
    # and a 10 year note, both maturing in 10 years.
    def calc_iri_convexity(self):
        
        return 1
    
    
class YieldCurve(object):
    
    def __init__(self):
        
        self.zero_rates = dict()
        self.instruments = dict()
        self.forward_rates_dict = dict()
        self.p=print
        self.logger = get_logger()
    
    def get_instruments_as_dict(self):
        
        return self.instruments
    
    def yield_curve_maturity_in_years(self,
                                      yield_curve_df):
        
        # list of maturities here is essentially the column names from the dataframe
        list_of_string_token_list = [c.split('_') for c in yield_curve_df.columns]
        yield_curve_usable_maturities = []
        for token_list in list_of_string_token_list:
            maturity_type = token_list[1]
            num_years = float(token_list[0])
            if (maturity_type == 'MO'):
                num_years = float(token_list[0])/12.0
            yield_curve_usable_maturities.append(num_years)
        #yield_curve_df.columns = [str('Yr_Maturity_Term_' + '%.3f' % c ).replace('.','') for c in sorted(yield_curve_usable_maturities)]
        yield_curve_df.columns = [str(c) for c in sorted(yield_curve_usable_maturities)]
        return yield_curve_df
            
    
    def add_instrument(self,
                       interest_rate_instrument):
        
        self.instruments[interest_rate_instrument.get_term()] = interest_rate_instrument
        
    def get_zero_rates(self):
        
        self.__bootstrap_zero_coupons__()
        self.__get_bond_spot_rates__()
        
        for iri in self.instruments.values():
            iri.set_zero_rate(self.zero_rates[iri.get_term()])
        
        return [self.zero_rates[term] for term in self.get_maturities()]
    
    def get_maturities(self):
        
        return sorted(self.instruments.keys())
    
    def get_instruments(self):
        
        return [self.instruments[key] for key in sorted(self.instruments.keys())]
    
    def __bootstrap_zero_coupons__(self):
        
        for term in self.instruments.keys():
            iri = self.instruments[term]
            if iri.get_coupon() == 0:
                self.zero_rates[term] = self.zero_coupon_spot_rate(iri)
                
    def __get_bond_spot_rates__(self):
        
        for term in self.get_maturities():
            iri = self.instruments[term]
            if iri.get_coupon() != 0:
                self.zero_rates[term] = self.__calculate_bond_spot_rate__(iri)
                
    def __calculate_bond_spot_rate__(self,
                                     iri):
        
        try:
            periods = iri.get_term() * iri.get_compounding_freq()
            value = iri.get_price()
            coupon_per_period = iri.get_coupon()/iri.get_compounding_freq()
            
            for i in range(int(periods)-1):
                t = (i+1)/float(iri.get_compounding_freq())
                spot_rate = self.zero_rates[t]
                discounted_coupon = coupon_per_period*np.exp(-spot_rate*t)
                value -= discounted_coupon
            last_period = int(periods)/float(iri.get_compounding_freq())
            spot_rate = -np.log(value/(iri.get_par_value()+coupon_per_period))/last_period
            #self.logger.info('interest_rates.YieldCurve.__calculate_bond_spot_rate__(): spot rate is %s', str(spot_rate))
            return spot_rate
        except:
            self.p("Error: spot rate not found for T=%s" % t)
            
                
    def zero_coupon_spot_rate(self,
                              iri):
        
        self.logger.info('interest_rates.YieldCurve.zero_coupon_spot_rate(): ParValue %s Price %s Term %s', str(iri.get_par_value()), str(iri.get_price()), str(iri.get_term()))
        return np.log(iri.get_par_value()/iri.get_price())/iri.get_term()
    
    #### the below methods are used in calculating forward rates from spot rates
    def __calculate_forward_rate__(self,
                                   iri1,
                                   iri2):
        
        R1 = iri1.get_zero_rate() # remember, spot rate = zero rate
        R2 = iri2.get_zero_rate()
        T1 = iri1.get_term()
        T2 = iri2.get_term()
        forward_rate = (R2*T2 - R1*T1)/(T2-T1)
        return forward_rate
    
    def calculate_forward_rates(self):
        
        iri_list = self.get_instruments()
        period_list = []
        for iri_term2, iri_term1 in zip(iri_list, iri_list[1:]):
            #self.p('iri_term1, iri_term2', iri_term1.get_term(), iri_term2.get_term())
            forward_rate = self.__calculate_forward_rate__(iri_term1, iri_term2)
            iri_term1.set_forward_rate(forward_rate)
            period_list.append(iri_term1.get_term())
            self.forward_rates_dict[iri_term1.get_term()] = forward_rate
        return self.forward_rates_dict
    
    def get_forward_rates(self):
        
        return self.forward_rates_dict
    
    # semi-annual compounding, NOT CONTINUOUS
    def get_spot_curve_from_par_curve(self):
        # this funciton is specifically for the treasury.gov data, which is par yield data.
        # example....
        # Maturity(Years) -- Par Yield -------- Spot Yield ----
        #      0.5             2.0000 %          2.0000 %
        #      1.0             2.4000 %          2.4024 %
        #      1.5             2.7600 %
        #      2.0             3.0840 %
        #      2.5             3.3756 %
        #      3.0             3.6380 %
        # the six month spot yield is the par yield at 6 months bc a 6 month bond only has 1 payment
        # toc computer 1 year spot yield, discount the payments on a 1 year par bond with annual coupon of 2.4%
        iri_obj_list = [self.instruments[iri_term] for iri_term in sorted(set(self.instruments.keys()))]
        logger.info("interest_rates.YieldCurve.get_spot_curve_from_par_curve(): iri_obj_list %s ", str(iri_obj_list))
            
        # bond equivalent yield, translated zero-coupon into bond equivalent, that is what par yield is
        # in addition, the market value of these yields are par value, since the yields are PAR YIELDS!
        # so the folloiwng is true for a par yield curve... bond equivalent yield = par yield = coupon
        price = iri_obj_list[0].get_price()
        par_value = iri_obj_list[0].get_par_value()
        if price != par_value:
            self.p("This function will FAIL because price and par value are not equal!!!")
        s=[]
        return_dict = dict()
        for i in range(0, len(iri_obj_list)):
            summer = 0
            par_value = iri_obj_list[i].get_par_value()
            for j in range(0, i):
                bey_i = iri_obj_list[i].get_coupon()/100.0
                t_j = iri_obj_list[j].get_term()
                cf = float(iri_obj_list[j].get_compounding_freq())
                summer += ((bey_i/cf)*par_value)/(1+s[j]/cf)**(t_j*cf)
            bey_i = iri_obj_list[i].get_coupon()/100.0
            t_i = iri_obj_list[i].get_term()
            cf = float(iri_obj_list[i].get_compounding_freq())
            numerator = par_value + par_value*(bey_i/cf)
            denominator = par_value - summer
            numden = numerator/denominator
            value = numden**(1/(t_i*cf))
            value -= 1
            value *=2.0
            return_dict[t_i] = value
            logger.info("interest_rates.YieldCurve.get_spot_curve_from_par_curve(): value %s %s %s %s ", str(value), str(bey_i), str(t_i), str(summer))
            s.append(value)
        
        self.p('s', s)
        return return_dict
        
class TimeValueMoney(object):
    
    bgn, end = 0, 1
    def __str__(self):
        
        return "n=%f, r=%f, pv=%f, pmt=%f, fv=%f" % ( self.n, self.r, self.pv, self.pmt, self.fv)
    
    def __init__(self,
                 n=0.0,
                 r=0.0,
                 pv=0.0,
                 pmt=0.0,
                 fv=0.0,
                 mode=end):
        
        self.p = print
        self.n = float(n)
        self.r = float(r)
        self.pv = float(pv)
        self.pmt = float(pmt)
        self.fv = float(fv)
        self.mode = mode
        
    def calculate_present_value(self):
        
        z = pow(1+self.r, -self.n)
        pva = self.pmt/self.r
        if (self.mode==TimeValueMoney.bgn):
            pva += self.pmt
        return -(self.fv*z + (1-z)*pva)
    
    def calculate_future_value(self):
        
        z = pow(1+self.r, -self.n)
        pva = self.pmt / self.r
        if (self.mode==TimeValueMoney.bgn):
            pva += self.pmt
        return -(self.pv + (1-z)*pva)/z
    
    def calculate_payment(self):
        z = pow(1+self.r, -self.n)
        if self.mode==TimeValueMoney.bgn:
            return (self.pv + self.fv*z) * self.r / (z-1) / (1+self.r)
        else:
            return (self.pv + self.fv*z) * self.r / (z-1)
    
    def calc_n(self):
        pva = self.pmt / self.r
        if (self.mode==TimeValueMoney.bgn): pva += self.pmt
        z = (-pva-self.pv) / (self.fv-pva)
        return -np.log(z) / np.log(1+self.r)
    
    def calculate_internal_rate_return(self):
        def function_fv(r, self):
            z = pow(1+r, -self.n)
            pva = self.pmt / r
            if (self.mode==TimeValueMoney.bgn): pva += self.pmt
            return -(self.pv + (1-z) * pva)/z
        return newton(f=function_fv, fArg=self, x0=.05, 
            y=self.fv, maxIter=1000, minError=0.0001)

from math import fabs
# f - function with 1 float returning float
# x0 - initial value
# y - desired value
# maxIter - max iterations
# minError - minimum error abs(f(x)-y)
def newton(f, fArg, x0, y, maxIter, minError):
    def func(f, fArg, x, y):
        return f(x, fArg) - y
    def slope(f, fArg, x, y):
        xp = x * 1.05
        return (func(f, fArg, xp, y)-func(f, fArg, x, y)) / (xp-x)      
    counter = 0
    while 1:
        sl = slope(f, fArg, x0, y);
        x0 = x0 - func(f, fArg, x0, y) / sl
        if (counter > maxIter): break
        if (fabs(f(x0, fArg)-y) < minError): break
        counter += 1
    return x0 
    
    
class VasicekModel(object):
    
    def __init__(self,
                 kappa,
                 sigma,
                 theta):
        
        self.kappa = kappa
        self.sigma = sigma
        self.theta = theta
        self.p = print
    
    # T is the period in number of years
    # N is the number of intervals for the modeling process
    def vasicek(self,
                r0, 
                T=1., 
                N=10, 
                seed=777):
    
        np.random.seed(seed)
        dt = T/float(N)
        rates = [r0]
        for i in range(N):
            dr = self.kappa*(self.theta-rates[-1])*dt + self.sigma*np.random.normal()
            rates.append(rates[-1] + dr)
        return range(N+1), rates

## CIR addresses the issue with negative rates in the Vasicek simulation 
class CoxIngersollRandModel(object):
    
    def __init__(self,
                 kappa,
                 sigma,
                 theta):
        
        self.kappa = kappa
        self.sigma = sigma
        self.theta = theta
        self.p = print
    
    # T is the period in number of years
    # N is the number of intervals for the modeling process
    def cir(self,
            r0,
            T=1,
            N=10,
            seed=777):
        
        np.random.seed(seed)
        dt = T/float(N)
        rates = [r0]
        for i in range(N):
            dr = self.kappa*(self.theta-rates[-1])*dt + self.sigma*sqrt(rates[-1])*np.random.normal()
            rates.append(rates[-1] + dr)
        return range(N+1), rates
    
class RendlemanBartterModel(object):
    
    def __init__(self,
                 theta,
                 sigma):
        
        self.theta = theta
        self.sigma = sigma
        self.p = print
    
    # lacks mean reversion - not built in to the model that IR's go back to long term mean eventually
    # this is why we see that long term average grows at t goes on.
    # geometric brownian motion simulation
    # like stock price stochastic process that is log normally distributed    
    def rendleman_bartter(self,
                          r0,
                          T=1.,
                          N=10,
                          seed=777):
        
        np.random.seed(seed)
        dt = T/float(N)
        rates = [r0]
        for i in range(N):
            dr = self.theta*rates[-1]*dt + self.sigma*rates[-1]*np.random.normal()
            rates.append(rates[-1] + dr)
        return range(N+1), rates

class BrennanSchwartz(object):
    
    # Two Factor Model
    # short rate reverts toward a long rate (the mean), where the long rate follows a stochastic process
    # this is another form of a geometric brownian motion
    
    # formula => dr(t) = Kappa*(theta-r(t))*d(t) + sigma*r(t)*weiner process(random normally distributed white noise)
    def __init__(self,
                 theta,
                 sigma,
                 kappa):

        self.theta = theta
        self.sigma = sigma
        self.kappa = kappa
        
    def brennan_schwartz(self,
                         r0,
                         T=1.,
                         N=10,
                         seed=777):
        
        np.random.seed(seed)
        dt = T/float(N)
        rates = [r0]
        for i in range(N):
            dr = self.kappa*(self.theta-rates[-1])*dt + self.sigma*rates[-1]*np.random.normal()
            rates.append(rates[-1] + dr)
        return range(N+1), rates
    
class EurodollarFuture(object):
    
    def __init__(self):
        
        self.expiration = 0
        self.imm_index = 100.0
        self.instr_yield = 0.0
        
    def calc_yield_from_IMM(self):
        
        return (100.0-self.imm_index)
    
    def get_yield(self):
        
        return self.instr_yield
    
    def set_yield(self,
                  instr_yield):
        
        self.instr_yield = instr_yield

class PiecewiseCubicSpline(object):
    
    def __init__(self,
                 yield_curve_dict):
        
        self.p=print
        self.yield_curve_dict = yield_curve_dict
        self.clean_yield_curve_tau_list = list(filter(lambda k: not np.isnan(self.yield_curve_dict[k].get_obs_ytm()), self.yield_curve_dict))
        self.clean_yield_curve_yield_list = [self.yield_curve_dict[clean_key].get_obs_ytm() for clean_key in sorted(set(self.clean_yield_curve_tau_list))]
        
    # cubic spline is simply an alternate method to Nelson Siegel
    def piecewise_cubic_spline(self):
        
        # break maturity term structure into 3 pieces
        # r0 is the short term rate, the y-intercept
        # r(t) = r0 + a*t + b*t^2 + c*t^3
        import scipy.interpolate as spi
        tr = self.clean_yield_curve_tau_list
        yr = self.clean_yield_curve_yield_list
        interp = spi.PchipInterpolator(tr, yr, extrapolate=True)
        return_dict = dict()
        for tau in np.linspace(0.5, 30.5, 61):
            value = float(interp(tau))
            return_dict[tau] = value
        return return_dict
    
    def plot_cubic_spline(self,
                          term_fitted_yield_dict):
        
        plot_interpolated_yields(term_fitted_yield_dict,
                                 self.clean_yield_curve_yield_list,
                                 self.clean_yield_curve_tau_list)

class NelsonSiegel(object):
    
    # yield curve dict has keys=taus, values=yields
    def __init__(self,
                 yield_curve_dict,
                 beta1=0.01,
                 beta2=0.01,
                 beta3=0.01,
                 gamma_ns=1.0,
                 beta4=0.01,
                 gamma_nss=1.0):
        
        self.p=print
        self.yield_curve_dict = yield_curve_dict
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.beta4 = beta4
        self.gamma_ns = gamma_ns
        self.gamma_nss = gamma_nss
        
    def reset_initial_params(self):
        
        self.beta1=0.01
        self.beta2=0.01
        self.beta3=0.01
        self.beta4=0.01
        self.gamma_ns=1.0
        self.gamma_nss=1.0
        
    def nelson_siegel(self,
                      tau):
        
        ns = self.beta1 + (self.beta2+self.beta3)*(self.gamma_ns/tau)*(1-np.exp(-tau/self.gamma_ns)) - self.beta3*np.exp(-tau/self.gamma_ns)
        return ns
    
    def nelson_siegel_svensson(self,
                               tau):
        
        # adds one additional term to the Neslon Siegel
        ns = self.nelson_siegel(tau)
        nss = ns + self.beta4*(self.gamma_nss/tau)*(1-np.exp(-tau/self.gamma_nss)) - self.beta4*np.exp(-tau/self.gamma_nss)
        return nss
    
    def calc_fmin(self,
                  gamma_ns,
                  gamma_s,
                  method='NSS',
                  minimizer='optimize.minimize.method=SLSQP',
                  cons=None,
                  bnds=None):
        
        clean_yield_curve_tau_list = list(self.yield_curve_dict.keys())
        clean_yield_curve_yield_list = list(self.yield_curve_dict.values())
        
        #clean_yield_curve_tau_list = list(filter(lambda k: not np.isnan(self.yield_curve_dict[k].get_obs_ytm()), self.yield_curve_dict))
        #clean_yield_curve_yield_list = [self.yield_curve_dict[clean_key].get_obs_ytm() for clean_key in sorted(set(clean_yield_curve_tau_list))]
        
        #pf_ns =  lambda bg, x: bg[0]+(bg[1]+bg[2])*(bg[3]/x)*(1-np.exp(-x/bg[3]))-bg[2]*np.exp(-x/bg[3])
        #pf_nss = lambda bg, x: bg[0]+(bg[1]+bg[2])*(bg[3]/x)*(1-np.exp(-x/bg[3]))-bg[2]*np.exp(-x/bg[3]) + bg[4]*(bg[5]/x)*(1-np.exp(-x/bg[5]))-bg[4]*np.exp(-x/bg[5])
        
        pf_ns =  lambda bg, x: bg[0]+(bg[1]+bg[2])*(gamma_ns/x)*(1-np.exp(-x/gamma_ns))-bg[2]*np.exp(-x/gamma_ns)
        pf_nss = lambda bg, x: bg[0]+(bg[1]+bg[2])*(gamma_ns/x)*(1-np.exp(-x/gamma_ns))-bg[2]*np.exp(-x/gamma_ns) + bg[3]*(gamma_s/x)*(1-np.exp(-x/gamma_s))-bg[3]*np.exp(-x/gamma_s)
        
        error_func_ns = lambda p, x, y: ((pf_ns(p,x)-y)**2).sum()
        error_func_nss = lambda p, x, y: ((pf_nss(p,x)-y)**2).sum()
        
        #p0_ns = np.array([self.beta1, self.beta2, self.beta3, self.gamma_ns])
        #p0_nss = np.array([self.beta1, self.beta2, self.beta3, self.gamma_ns, self.beta4, self.gamma_nss])
        
        p0_ns = np.array([self.beta1, self.beta2, self.beta3])
        p0_nss = np.array([self.beta1, self.beta2, self.beta3, self.beta4])
        
        #cons = ({'type':'ineq',
        #         'fun' : lambda bg: np.array([bg[0] + bg[1]])})
        
        x=np.array(clean_yield_curve_tau_list) # these are the maturities, they should be in years, so 1 month would be 1/12 or .083
        y=np.array(clean_yield_curve_yield_list) # these are the actual yields that correspond for the above maturities
        
        # fit the data with fmin from scipy
        if method=='NS':
            if minimizer=='optimize.fmin':
                p = optimize.fmin(error_func_ns, p0_ns, args=(x,y))
            elif minimizer=='optimize.minimize.method=SLSQP':
                p = optimize.minimize(error_func_ns, p0_ns, args=(x,y), method='SLSQP', bounds=bnds, constraints=cons)
            elif minimizer=='optimize.minimize.method=Nelder-Mead':
                p = optimize.minimize(error_func_ns, p0_ns, args=(x,y), method='Nelder-Mead')
            elif minimizer=='optimize.minimize.method=Powell':
                p = optimize.minimize(error_func_ns, p0_ns, args=(x,y), method='Powell')
            elif minimizer=='optimize.minimize.method=COBYLA':
                p = optimize.minimize(error_func_ns, p0_ns, args=(x,y), method='COBYLA')
            elif minimizer=='optimize.minimize.method=Newton-CG':
                p = optimize.minimize(error_func_ns , p0_ns, args=(x,y), method='Newton-CG')
        elif method=='NSS':
            if minimizer=='optimize.fmin':
                p = optimize.fmin(error_func_nss, p0_nss, args=(x,y))
            elif minimizer=='optimize.minimize.method=SLSQP':
                cons = ({'type':'ineq',
                         'fun' : lambda bgp: np.array([bgp[0] + bgp[1]]) } )
                b = (0.0, 0.5)
                no_b = (-np.inf, np.inf)
                bnds = (b,no_b,no_b,no_b)
                p = optimize.minimize(error_func_nss, p0_nss, args=(x,y), method='SLSQP', bounds=bnds, constraints=cons)
            elif minimizer=='optimize.minimize.method=Nelder-Mead':
                p = optimize.minimize(error_func_nss, p0_nss, args=(x,y), method='Nelder-Mead')
            elif minimizer=='optimize.minimize.method=Powell':
                p = optimize.minimize(error_func_nss, p0_nss, args=(x,y), method='Powell')
            elif minimizer=='optimize.minimize.method=COBYLA':
                self.p('minimizing with COBYLA...')
                p = optimize.minimize(error_func_nss, p0_nss, args=(x,y), method='COBYLA', options={'maxiter':7})
            elif minimizer=='optimize.minimize.method=Newton-CG':
                p = optimize.minimize(error_func_nss, p0_nss, args=(x,y), method='Newton-CG', jac=None)
        bg = p
        # we will extend out 1 year from the 30 year bond
        return_dict = dict()
        return_dict['optimizer_message'] = bg.message
        return_dict['number_of_evals'] = bg.nfev
        return_dict['value_of_min_func'] = bg.fun
        return_dict['yield_curve'] = dict()
        for tau in np.linspace(0.5,30.5, 61):
            fitted_yield = -1000.0
            if method=='NS':
                self.beta1 = bg.x[0]
                self.beta2 = bg.x[1]
                self.beta3 = bg.x[2]
                #self.gamma_ns = bg.x[3]
                return_dict['yield_curve'][tau] = self.nelson_siegel(tau)
            elif method=='NSS':
                self.beta1 = bg.x[0]
                self.beta2 = bg.x[1]
                self.beta3 = bg.x[2]
                #self.gamma_ns = bg.x[3]
                self.beta4 = bg.x[3]
                #self.gamma_nss = bg.x[5]
                return_dict['yield_curve'][tau] = self.nelson_siegel_svensson(tau)
            #return_dict[tau] = fitted_yield
        #self.p(return_dict)
        return return_dict
        # the dictionary returned here is keys=maturities (in years), value=fitted yield
        # fitted yields are fitted to the available yield curve, where the OLS function is minimized.

# these are the actual current rates of LIBOR for different duration of loan
class LiborCurve(object):
    
    def __init__(self,
                 libor_curve):
        
        self.libor_curve = libor_curve
        
    def calculate_IFR(self,
                      term_D1,
                      next_term_D2,
                      days_into_future_start_date_D3,
                      R1,
                      R2):
        
        # ie. 90 day term, rate is R1
        #     180 day term, rate is R2
        #     term starts D3 days from now
        numerator = (1+R2*(next_term_D2/360.0))
        denominator = (days_into_future_start_date_D3/360.0)*(1+R1*(term_D1/360.0))
        IFR = (numerator/denominator) - (1/(days_into_future_start_date_D3/360.0))
        return IFR     
        
class EurodollarFutureCurve(object):
    
    def __init__(self):
        
        self.ed_curve = dict()
        
    def get_ed_curve(self):
        
        return self.ed_curve
    
from mpl_toolkits.mplot3d import Axes3D
import copy as copylib
#from progressbar import *
import pandas as pd
#pylab.rcParams['figure.figsize'] = (16,4.5)

from root.nested import get_logger

class HeathJarrowMortonModel(object):
    
    def __init__(self,
                 yield_curve_df):
        
        self.logger = get_logger()
        
        self.p = print
        self.yield_curve_df = yield_curve_df/100.0
        self.all_counter = 0
        self.Nelder_Mead_min_func_summer = 0.0
        self.Powell_min_func_summer = 0.0
        self.SLSQP_min_func_summer = 0.0
        self.COBYLA_min_func_summer = 0.0
        self.Newton_CG_min_func_summer = 0.0
        
        self.MFE_EXCEED_MSG = 'Maximum number of function evaluations has been exceeded.'
        
        self.Nelder_Mead_mfe_exceed_min_func_summer=0.0
        self.Nelder_Mead_mfe_exceed_counter=0
        self.Powell_mfe_exceed_min_func_summer=0.0
        self.Powell_mfe_exceed_counter=0
        self.SLSQP_mfe_exceed_min_func_summer = 0.0
        self.SLSQP_mfe_exceed_counter=0
        self.COBYLA_mfe_exceed_min_func_summer = 0.0
        self.COBYLA_mfe_exceed_counter=0
        self.Newton_mfe_exceed_min_func_summer = 0.0
        self.Newton_mfe_exceed_counter=0
    
    def calc_fmin_stats(self):
        
        self.logger.info('total number of rows %s', str(self.all_counter))
        
        self.logger.info('Nelder-Mead avg of all min_func %s', str(self.Nelder_Mead_min_func_summer/self.all_counter))
        self.logger.info('Powell avg of all min func %s', str(self.Powell_min_func_summer/self.all_counter))
        self.logger.info('SLSQP avg of all min func %s', str(self.SLSQP_min_func_summer/self.all_counter))
        self.logger.info('COBYLA avg of all min func %s', str(self.COBYLA_min_func_summer/self.all_counter))
        self.logger.info('Newton avg of all min func %s', str(self.Newton_CG_min_func_summer/self.all_counter))
        
        self.logger.info('Nelder-Mead MFE-Exceed avg min func %s', str(self.Nelder_Mead_mfe_exceed_min_func_summer/self.Nelder_Mead_mfe_exceed_counter))
        #self.logger.info('Powell MFE-Exceed avg min func %s', str(self.Powell_mfe_exceed_min_func_summer/self.Powell_mfe_exceed_counter))
        #self.logger.info('SLSQP MFE-Exceed avg min func %s', str(self.SLSQP_mfe_exceed_min_func_summer/self.SLSQP_mfe_exceed_counter))
        #self.logger.info('COBYLA MFE-Exceed avg min func %s', str(self.COBYLA_mfe_exceed_min_func_summer/self.COBYLA_mfe_exceed_counter))
        #self.logger.info('Newton MFE-Exceed avg min func %s', str(self.Newton_mfe_exceed_min_func_summer/self.Newton_mfe_exceed_counter))
         
    def get_df_info(self):
        
        self.yield_curve_df.info()
        self.yield_curve_df.describe()
        pd.options.display.max_rows=10
        #self.p(self.yield_curve_df.head())
        
    def create_yield_curve_obj_dict(self):
        
        yield_curve_obj = YieldCurve()
        _ = yield_curve_obj.yield_curve_maturity_in_years(self.yield_curve_df)
        columns = self.yield_curve_df.columns
        self.p(columns)
        dict_of_yield_curve_objects = dict()
        for row in self.yield_curve_df.itertuples():
            for i in range(1, len(columns) +1):
                obs_ytm = row[i]
                term_yrs = columns[i-1]
                #self.p(obs_ytm, term_yrs)
                iri = InterestRateInstrument(100,
                                             term_yrs,
                                             0.0,
                                             obs_ytm=obs_ytm)
                ## now, get the price from the obs_YTM
                iri_price = iri.get_price_from_ytm(obs_ytm)
                iri.set_price(iri_price)
                self.yield_curve_obj.add_instrument(iri)
            ## anything to do before moving on to the next row???
            dict_of_yield_curve_objects[row[0]] = yield_curve_obj
        
        return dict_of_yield_curve_objects

    def vectorized_create_yield_curve(self):
        
        series_of_yield_curve_objs = self.yield_curve_df.apply(create_yield_curve_obj, axis=1)
        self.yield_curve_df['Yield_Curve'] = series_of_yield_curve_objs
        return self.yield_curve_df
        
    def do_interpolation(self,
                         interp_method='PCS'):
        
        # loop thru each date in the dataframe, or use vectorized approach
        if interp_method=='PCS':
            series_interp_yield_curves = self.yield_curve_df.apply(self.PCS_interpolation, axis=1)
            self.yield_curve_df['PCS_Interp_Yield_Curve'] = series_interp_yield_curves
        elif interp_method=='NSS':
            series_interp_yield_curves = self.yield_curve_df.apply(self.NS_S_interpolation, axis=1)
            # what we return here is a lot more than just a yield curve
            self.yield_curve_df = pd.concat([self.yield_curve_df, series_interp_yield_curves], axis=1)
            self.yield_curve_df.info()
            self.yield_curve_df.describe()
            
        return self.yield_curve_df
    
    def do_bootstrap_spot(self):
        
        # semi-annual compounding only being implemented
        # TODO: continuous compounding.
        series_of_curves = self.yield_curve_df.apply(self.bootstrap_spot_yield_curve, axis=1)
        #self.yield_curve_df['Spot_Yield_Curve'] = series_of_spot_curves
        self.yield_curve_df = pd.concat([self.yield_curve_df, series_of_curves], axis=1)
        self.yield_curve_df.info()
        self.yield_curve_df.describe()
        return self.yield_curve_df
    
    def PCS_interpolation(self,
                          row):
        
        yco = row.Yield_Curve
        yield_curve_dict = yco.get_instruments_as_dict()
        pcs = PiecewiseCubicSpline(yield_curve_dict)
        interp_yield_curve_dict = pcs.piecewise_cubic_spline()
        return interp_yield_curve_dict
    
    def bootstrap_spot_yield_curve(self,
                                   row):
        
        pcs_interp_yield_curve_dict = row.PCS_Interp_Yield_Curve
        #self.logger.info('interest_rates.HeathJarrowMortonModel.bootstrap_spot_yield_curve(): pcs_interp_yield_curve_dict %s %s', str(row.name), str(pcs_interp_yield_curve_dict))
        yield_curve_obj = YieldCurve()
        interp_yield_list = []
        for tau in sorted(set(pcs_interp_yield_curve_dict.keys())):
            interp_yield = pcs_interp_yield_curve_dict[tau]
            interp_yield_list.append(interp_yield) 
            # create an Interest Rate Instrument object
            iri_par_yield = InterestRateInstrument(100,
                                                   tau,
                                                   interp_yield*100.0,
                                                   100)
            yield_curve_obj.add_instrument(iri_par_yield)
        y = yield_curve_obj.get_zero_rates()
        x = yield_curve_obj.get_maturities()
        # convert the continuous rates to semi-annual compounding
        # Rm = m*(exp(Rc/m) -1)
        # lambda k: not np.isnan(yield_curve_dict[k].get_obs_ytm()), yield_curve_dict
        get_semi_annual_cmpd_rate = list(map(lambda k: 2*(np.exp(k/2) - 1), y))
        return_dict = dict(zip(x,get_semi_annual_cmpd_rate))
        
        # filter a dict, using a list of maturities
        original_maturities = [0.083, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]
        orig_dict_from_ret_dict = {k:v for (k,v) in return_dict.items() if k in original_maturities }
        #self.logger.info('interest_rates.HeathJarrowMortonModel.bootstrap_spot_yield_curve(): return_dict %s ', str(return_dict))
        
        #=======================================================================
        # if np.datetime64(str(row.name).split('T')[0]) == np.datetime64('2002-02-26'):
        #     import matplotlib.pyplot as plt
        #     plt.plot(x, np.array(get_semi_annual_cmpd_rate), 'ro')
        #     plt.plot(x, np.array(interp_yield_list), 'b')
        #     plt.title('Cubic Spline Interpolation vs. Spot Curve')
        #     plt.ylabel('term')
        #     plt.xlabel('Yield')
        #     plt.show()
        #=======================================================================
        
        return_series = self.NS_S_interpolation(orig_dict_from_ret_dict,
                                                pcs_interp_yield_curve_dict,
                                                row.name)
        
        return return_series
    
    def NS_S_interpolation(self,
                           spot_yco_dict,
                           pcs_interp_yield_curve_dict,
                           date_of_yield_curve):
        logger_loc = 'interst_rates.HeathJarrowMortonModel.NS_S_interpolation(): '
                   
        #self.logger.info(logger_loc + ' row.name %s ', str(date_of_yield_curve))
        # this dictionary contains spot rates of PCS interpolated yield curve
        # run the NSS only on the original data points, spot rates without the interpolated coordinates
        
        nss = NelsonSiegel(yield_curve_dict=spot_yco_dict)
        lowest_min_func_value = 1000000.0
        lowest_min_func_return_dict= None
        for gamma_ns in np.linspace(0.25,4.0,16):
            for gamma_s in np.linspace(4,0.25,16):
                if (abs(gamma_ns - gamma_s) < 1.0):
                    continue
                else:
                    returned_min_params_SLSQP = nss.calc_fmin(gamma_ns, gamma_s, minimizer='optimize.minimize.method=SLSQP')
                    value_of_min_func = returned_min_params_SLSQP['value_of_min_func']
                    if int(value_of_min_func*1000000.0) < int(lowest_min_func_value*1000000.0):
                        lowest_min_func_value = value_of_min_func
                        lowest_min_func_return_dict = returned_min_params_SLSQP
                        nss.gamma_ns = gamma_ns
                        nss.gamma_nss = gamma_s
                #self.logger.info('nss.beta1,nss.beta2,gamma_ns,gamma_s %s, %s, %s %s', nss.beta1, nss.beta2, gamma_ns, gamma_s)
        #self.logger.info(logger_loc + 'nss.beta1, nss.beta2, nss.beta1-nss.beta2 %s %s %s', str(nss.beta1), str(date_of_yield_curve), str(nss.beta1+nss.beta2))
        self.logger.info(logger_loc + ' nss.beta1, bss.beta2, nss.beta3, nss.beta4, gamma_ns, gamma_s %s, %s, %s, %s, %s, %s %s', str(date_of_yield_curve),\
                        nss.beta1, nss.beta2, nss.beta3, nss.beta4, nss.gamma_ns, nss.gamma_nss)
        #return_dict['optimize.minimize.method=Nelder-Mead,Betas'] = {1: nss.beta1, 2:nss.beta2, 3:nss.beta3, 4: nss.beta4}
        #return_dict['optimize.minimize.method=Nelder-Mead,Gammas'] = {1: nss.gamma_ns, 2: nss.gamma_nss}
        #return_dict['optimize.minimize.method=Nelder-Mead,Yield_Curve'] = returned_min_params_Nelder_Mead['yield_curve']
        #=======================================================================
        # if nss.beta1 > 1.0:
        #     import matplotlib.pyplot as plt
        #     x = np.array(list(return_dict['optimize.minimize.method=Nelder-Mead,Betas'].keys()))
        #     y = np.array(list(return_dict['optimize.minimize.method=Nelder-Mead,Betas'].values()))
        #     plt.plot(x, y, 'r')
        #     plt.title('Betas Parameters')
        #     plt.ylabel('Betas')
        #     plt.xlabel('param #')
        #     plt.show()
        #  
        #     x = np.array(list(return_dict['optimize.minimize.method=Nelder-Mead,Gammas'].keys()))
        #     y = np.array(list(return_dict['optimize.minimize.method=Nelder-Mead,Gammas'].values()))
        #     plt.plot(x, y, 'r')
        #     plt.title('Gammas Parameters')
        #     plt.ylabel('Gammas')
        #     plt.xlabel('param #')
        #     plt.show()
        #  
        #     x_yc_nm = np.array(list(returned_min_params_Nelder_Mead['yield_curve'].keys()))
        #     y_yc_nm = np.array(list(returned_min_params_Nelder_Mead['yield_curve'].values()))
        #     x_yc_spot = np.array(list(spot_yco_dict.keys()))
        #     y_yc_spot = np.array(list(spot_yco_dict.values()))
        #     x_yc_pcs_interp = np.array(list(pcs_interp_yield_curve_dict.keys()))
        #     y_yc_pcs_interp = np.array(list(pcs_interp_yield_curve_dict.values()))
        #     plt.plot(x_yc_nm, y_yc_nm, 'r')
        #     plt.plot(x_yc_spot, y_yc_spot, 'g')
        #     plt.plot(x_yc_pcs_interp, y_yc_pcs_interp, 'b')
        #     plt.title('Actual NSS Yield Curve')
        #     plt.xlabel('Yield')
        #     plt.ylabel('tau')
        #     plt.show()
        #=======================================================================
            
        
        
        return_series = pd.Series(data=[nss.beta1, nss.beta2, nss.beta3, nss.beta4, nss.gamma_ns, nss.gamma_nss, nss.beta1+nss.beta2,abs(nss.gamma_ns-nss.gamma_nss),\
                                        lowest_min_func_return_dict['yield_curve']], name=date_of_yield_curve,\
                                        index=['Beta1', 'Beta2', 'Beta3', 'Beta4', 'Gamma1', 'Gamma2', 'Beta1_Plus_Beta2', 'Abs_Gamma1_Minus_Gamma2', 'Yield_Curve'])
        
        #nss.reset_initial_params()
        #returned_min_params_SLSQP = nss.calc_fmin()
        #return_dict['optimize.minimize.method=SLSQP,Betas'] = {1: nss.beta1, 2: nss.beta2, 3: nss.beta3, 4:nss.beta4}
        #return_dict['optimize.minimize.method=SLSQP,Gammas'] = {1: nss.gamma_ns, 2: nss.gamma_nss}
        #return_dict['optimize.minimize.method=SLSQP,Yield_Curve'] = returned_min_params_SLSQP['yield_curve']
        
        #self.Nelder_Mead_min_func_summer += returned_min_params_Nelder_Mead['value_of_min_func']
        #self.SLSQP_min_func_summer += returned_min_params_SLSQP['value_of_min_func']
        self.all_counter +=1
        
        #if returned_min_params_Nelder_Mead['optimizer_message'] == self.MFE_EXCEED_MSG:
        #    mfe_exceed_min_func = returned_min_params_Nelder_Mead['value_of_min_func']
        #    self.Nelder_Mead_mfe_exceed_min_func_summer += mfe_exceed_min_func
        #    self.Nelder_Mead_mfe_exceed_counter+=1
        
        #elif returned_min_params_SLSQP['optimizer_message'] == self.MFE_EXCEED_MSG:
        #    mfe_exceed_min_func = returned_min_params_SLSQP['value_of_min_func']
        #    self.SLSQP_mfe_exceed_min_func_summer += mfe_exceed_min_func
        #    self.SLSQP_exceed_counter+=1
            
        #return_dict['optimize.minimize.method=SLSQP'] = returned_min_params_SLSQP
        #return_dict['optimize.minimize.method=Nelder-Mead'] = returned_min_params_Nelder_Mead
        
        #=======================================================================
        return return_series
    
    def do_NSS_params_analysis(self):
        
        import matplotlib.pyplot as plt
        #for i in range(6):
        plt.figure(1)
        self.yield_curve_df.Beta1.plot()
        plt.title("Beta1")
        plt.figure(2)
        self.yield_curve_df.Beta2.plot()
        plt.figure(3)
        self.yield_curve_df.Beta3.plot()
        plt.figure(4)
        self.yield_curve_df.Beta4.plot()
        plt.figure(5)
        self.yield_curve_df.Beta1_Plus_Beta2.plot()
        plt.figure(6)
        self.yield_curve_df.Gamma1.plot()
        plt.figure(7)
        self.yield_curve_df.Gamma2.plot()
        plt.figure(8)
        self.yield_curve_df.Abs_Gamma1_Minus_Gamma2.plot()
        plt.show()
        
def plot_interpolated_yields(term_fitted_yield_dict,
                             clean_yield_curve_yield_list,
                             clean_yield_curve_tau_list):
    
    y_interp = np.array([term_fitted_yield_dict[term] for term in sorted(set(term_fitted_yield_dict.keys()))])
    x_interp = np.array([term for term in sorted(set(term_fitted_yield_dict.keys()))])
    
    x = np.array(clean_yield_curve_tau_list)
    y = np.array(clean_yield_curve_yield_list)
    
    import matplotlib.pyplot as plt
    plt.plot(x, y)
    plt.plot(x_interp, y_interp)
    plt.title('Cubic Spline Interpolation')
    plt.ylabel('term')
    plt.xlabel('Yield')
    plt.show()

# old function, do not use for NSS interpolation      
def do_NS_S_interpolation(row, row_counter, min_func_summer):
    
    yco = row.Yield_Curve
    # need to pass in a dictionary, key = term, value = rate
    # at this point, we are still dealing with par yields
    yield_curve_dict = yco.get_instruments_as_dict()
    #print('do_NS_S_interplation.yield_curve_dict ', yield_curve_dict)
    
    clean_yield_curve_tau_list = list(filter(lambda k: not np.isnan(yield_curve_dict[k].get_obs_ytm()), yield_curve_dict))
    clean_yield_curve_yield_list = [yield_curve_dict[clean_key].get_obs_ytm() for clean_key in sorted(set(clean_yield_curve_tau_list))]
    
    nss = NelsonSiegel(yield_curve_dict=yield_curve_dict)
    returned_min_params = nss.calc_fmin(minimizer='optimize.minimize.method=Nelder-Mead')
    #print('returned min params', returned_min_params, row.name)
    
    if returned_min_params['optimizer_message'] == 'Maximum number of function evaluations has been exceeded.':
        min_func = returned_min_params['value_of_min_func']
        
        # create the fitted yield curve
        term_fitted_yield_dict = dict()
        for term in np.linspace(0.5,30.5, 61):
            nss_fitted_yield = returned_min_params[term]
            term_fitted_yield_dict[term] = nss_fitted_yield
        y_fitted = np.array([term_fitted_yield_dict[term] for term in sorted(set(term_fitted_yield_dict.keys()))])
        x_fitted = np.array([term for term in sorted(set(term_fitted_yield_dict.keys()))])
        y=np.array(clean_yield_curve_yield_list)
        x=np.array(clean_yield_curve_tau_list)
        
        import matplotlib.pyplot as plt
        plt.plot(x, y*100.0)
        plt.plot(x_fitted, y_fitted*100.0)
        plt.title("NSS fitted vs. Raw Par Yield")
        plt.ylabel("Par Yields")
        plt.xlabel("Maturity in Years")
        plt.show()
    
    nss.reset_initial_params()
        
def create_forward_rates_from_ytm(yco):
    
    yco_with_frs = yco.calculate_forward_rates()
    #print(yco_with_frs)
    return yco_with_frs
    
def create_yield_curve_obj(row):
    
    iri = None
    yield_curve_obj = YieldCurve()
    for col_name in row.index:
        token_list = str(col_name).split('_')
        maturity_type = token_list[1]
        num_years = float(token_list[0])
        if (maturity_type == 'MO'):
            num_years = float(token_list[0])/12.0
        term_yrs = num_years
        obs_ytm = row[col_name]
        iri = InterestRateInstrument(100,
                                     term_yrs,
                                     0.0,
                                     obs_ytm=obs_ytm)
        iri.set_zero_rate(obs_ytm)
        # now get the price from the obs_ytm
        iri_price = iri.get_price_from_ytm(obs_ytm)
        iri.set_price(iri_price)
        yield_curve_obj.add_instrument(iri)
         
    return yield_curve_obj
                     
if __name__ == "__main__":
    p=print
    
    # Mortgage payment
    pmt = TimeValueMoney(n=25*12, r=.04/12, pv=500000, fv=0).calculate_payment()
    p("Payment = %f" % pmt)
    
    # Yield to Maturity, semi annual bond, par value $100, 6% coupon, 10 years to maturity, current market price $80
    irr = 2*TimeValueMoney(n=10*2, pmt=6/2, pv=-80, fv=100).calculate_internal_rate_return()
    p("Yield to Maturity interest rate = %f" % irr) 
    
    # Arbitrage Free Price of Bond, Market Interest Rate is 6%, 8 years until maturity, coupon rate is 5%, par value is $100
    present_value_bond = TimeValueMoney(r=.06, n=8, pmt=5, fv=100).calculate_present_value()
    p("Price of bond should be %f" % present_value_bond)
    
    # Arbitrage Free Price of Bond, Market Interest Rate is 6%, 8 years until maturity, coupon rate is 0%, par value is $100
    present_value_bond = TimeValueMoney(r=.008, n=.25, pmt=0, fv=1).calculate_present_value()
    p("Price of bond should be %f" % present_value_bond)
    
    present_value_t_bill = TimeValueMoney(r=0.015, n=0.5, pmt=0, fv=1000).calculate_present_value()
    p("Price of 180 day tbill should be %f" % present_value_t_bill)
    
    real_treasury_par_yield_curve = YieldCurve()
    jan_second_2018_yields = [1.29,1.44,1.61,1.83,1.92,2.01,2.25,2.38,2.46,2.64,2.81]
    maturities = [1.0/12.0, 3.0/12.0, 6.0,12.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]
    for index in range(0,len(jan_second_2018_yields)):
        t_y = jan_second_2018_yields[index]
        mat = maturities[index]
        iri = InterestRateInstrument(1000,
                                     mat,
                                     t_y,
                                     1000)
        real_treasury_par_yield_curve.add_instrument(iri)
    y = real_treasury_par_yield_curve.get_spot_curve_from_par_curve()
    p('testing get SPOT yield curve from PAR yield curve')
    p(y)
    
    par_yield_curve = YieldCurve()
    iri_par_yield_6_mos_tbill = InterestRateInstrument(100,
                                                       0.5,
                                                       2.0,
                                                       100)
    iri_par_yield_12_mos_tbill = InterestRateInstrument(100,
                                                        1.0,
                                                        2.4,
                                                        100)
    iri_par_yield_18_mos_tbill = InterestRateInstrument(100,
                                                        1.5,
                                                        2.76,
                                                        100)
    iri_par_yield_24_mos_tbill = InterestRateInstrument(100,
                                                        2.0,
                                                        3.084,
                                                        100)
    iri_par_yield_30_mos_tbill = InterestRateInstrument(100,
                                                        2.5,
                                                        3.3756,
                                                        100)
    iri_par_yield_36_mos_tbill = InterestRateInstrument(100,
                                                        3.0,
                                                        3.638,
                                                        100)
    par_yield_curve.add_instrument(iri_par_yield_6_mos_tbill)
    par_yield_curve.add_instrument(iri_par_yield_12_mos_tbill)
    par_yield_curve.add_instrument(iri_par_yield_18_mos_tbill)
    par_yield_curve.add_instrument(iri_par_yield_24_mos_tbill)
    par_yield_curve.add_instrument(iri_par_yield_30_mos_tbill)
    par_yield_curve.add_instrument(iri_par_yield_36_mos_tbill)
    
    y = par_yield_curve.get_zero_rates()
    x = par_yield_curve.get_maturities()
    p(y)
    p(x)
    
    # convert the continuous rates to semi-annual compounding
    # Rm = m*(exp(Rc/m) -1)
    # lambda k: not np.isnan(yield_curve_dict[k].get_obs_ytm()), yield_curve_dict
    get_semi_annual_cmpd_rate = list(map(lambda k: 2*(np.exp(k/2) - 1), y))
    p('get semi annual cmpd rate', get_semi_annual_cmpd_rate)
    
    y = par_yield_curve.get_spot_curve_from_par_curve()
    p('testing get SPOT yield curve from PAR yield curve')
    p(y)
    
    yield_curve = YieldCurve()
    iri_3_mmth_tbill = InterestRateInstrument(100,
                                              0.25,
                                              0,
                                              97.5)
    iri_6_mnth_tbill = InterestRateInstrument(100,
                                              0.5,
                                              0,
                                              94.9)
    iri_12_mnth_tbill = InterestRateInstrument(100,
                                               1.0,
                                               0,
                                               90.0)
    iri_18_mnth_tbill = InterestRateInstrument(100,
                                               1.5,
                                               8.0,
                                               96.0)
    iri_24_mnth_tbill = InterestRateInstrument(100,
                                               2.0,
                                               12.0,
                                               101.6)
    yield_curve.add_instrument(iri_3_mmth_tbill)
    yield_curve.add_instrument(iri_6_mnth_tbill)
    yield_curve.add_instrument(iri_12_mnth_tbill)
    yield_curve.add_instrument(iri_18_mnth_tbill)
    yield_curve.add_instrument(iri_24_mnth_tbill)
    
    # for a 1 year bond ( with maturity in 1 year), with a price equal to par, the spot(zero) rate is the coupon rate 
    
    y = yield_curve.get_zero_rates()
    x = yield_curve.get_maturities()
    p(y)
    p(x)
    
    import matplotlib.pyplot as plt
    plt.plot(x, y)
    plt.title("zero curve")
    plt.ylabel("Zero Rate (%/100)")
    plt.xlabel("Maturity in Years")
    plt.show()
    
    iri_1_year_gilt = InterestRateInstrument(100,
                                             1.0,
                                             3.0,
                                             100.0,
                                             compounding_freq=1)
    iri_2_year_gilt = InterestRateInstrument(100,
                                             2.0,
                                             5.0,
                                             100.0,
                                             compounding_freq=1)
    iri_3_year_gilt = InterestRateInstrument(100,
                                             3.0,
                                             7.0,
                                             100.0,
                                             compounding_freq=1)
    yield_curve_gilt = YieldCurve()
    yield_curve_gilt.add_instrument(iri_1_year_gilt)
    yield_curve_gilt.add_instrument(iri_2_year_gilt)
    yield_curve_gilt.add_instrument(iri_3_year_gilt)
    y_gilt = yield_curve_gilt.get_zero_rates()
    x_gilt = yield_curve_gilt.get_maturities()
    p('gilt', y_gilt)
    p('gilt', x_gilt)
    plt.plot(x_gilt, y_gilt)
    plt.title("zero curve Gilts, 1,2,3 year maturity")
    plt.ylabel("Zero rate (%/100)")
    plt.xlabel("Maturity in Years")
    plt.show()
    #===========================================================================
    # iri_3_mmth_tbill.set_zero_rate(3.0)
    # iri_3_mmth_tbill.set_term(1.0)
    # iri_6_mnth_tbill.set_zero_rate(4.0)
    # iri_6_mnth_tbill.set_term(2.0)
    # iri_12_mnth_tbill.set_zero_rate(4.6)
    # iri_12_mnth_tbill.set_term(3.0)
    # iri_18_mnth_tbill.set_zero_rate(5.0)
    # iri_18_mnth_tbill.set_term(4.0)
    # iri_24_mnth_tbill.set_zero_rate(5.3)
    # iri_24_mnth_tbill.set_term(5.0)
    #===========================================================================
    
    p(yield_curve.calculate_forward_rates())
    plt.plot(yield_curve.get_forward_rates().keys(), yield_curve.get_forward_rates().values())
    plt.title("forward rates curve")
    plt.ylabel("forward rate (%/100)")
    plt.xlabel("Year")
    plt.show()
        
    vasicek_obj = VasicekModel(0.2,
                               0.012,
                               0.01)
    v_x, v_y = vasicek_obj.vasicek(0.01875, T=10, N=200)
    p(v_x, v_y)
    plt.title('Vasicek r0=1.875%, Theta=1%, Sigma=1.2%, Kappa=2%')
    plt.plot(v_x, v_y)
    plt.show()
        
    cir_obj = CoxIngersollRandModel(0.2, #kappa
                                    0.012, #sigma
                                    0.01) #theta
    cir_x, cir_y = cir_obj.cir(0.01875, T=10, N=200)
    p(cir_x,cir_y)
    plt.title('CIR r0=1.875%, Theta=1%, Sigma=1.2%, Kappa=2%')
    plt.plot(cir_x, cir_y)
    plt.show()
    
    rb_obj = RendlemanBartterModel(0.01,
                                   0.012)
    rb_x, rb_y = rb_obj.rendleman_bartter(0.01875, 10, 200)
    plt.title('Rendleman Bartter r0=1.875%, Theta=1%, Sigma=1.2%')
    plt.plot(rb_x, rb_y)
    plt.show()
    
    bs_obj = BrennanSchwartz(0.01,
                             0.012,
                             0.2)
    
    x_bs, y_bs = bs_obj.brennan_schwartz(0.01875, 10., 10000)
    plt.title('Brennan Schwartz r0=1.875%, Theta=1%, Sigma=1.2%, Kappa=2%')
    plt.plot(x_bs, y_bs)
    plt.show()
    
    iri_obj = InterestRateInstrument(100.0, 1.5, 5.75, 95.0428, compounding_freq=2)
    ytm_iri_obj = iri_obj.get_yield_to_maturity()
    iri_obj.set_calculated_ytm(ytm_iri_obj)
    p('ytm is ', ytm_iri_obj)
    
    iri_price = iri_obj.get_price_from_ytm(ytm_iri_obj)
    p('bond price', iri_price)
    