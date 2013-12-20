"""
    Generic Delay and Event Distributions (time_count_distributions_generic)
    ========================================================================
    Classes for 'generic' (i.e., non-source-specific) distributions of travel time, delay or event counts.

    Each class includes a ``prob()`` function and either ``max_delay()`` and ``min_delay()`` functions,
    or ``max_count()`` and ``min_count()`` functions.

        *   *Arbitrary Distribution* - can represent second-by-second probabilities for
            any travel time component, including travel time reductions (negative delay)
        *   *Normal Distribution* - represents probabilities for normally-distributed
            travel time components (or subcomponents), only for positive delay values
            (negative portions of the distribution are discarded and the distribution is
            rescaled accordingly)
        *   *Lognormal Distribution* - represents probabilities for lognormally-distributed
            travel time components (or subcomponents), only for positive delay values
            (negative portions of the distribution are discarded and the distribution is
            rescaled accordingly). Two constructors are available (mean and s.d. of
            lognormal variable, or mean and s.d. of normal dist. of which this variable
            is log-distributed).
        *   *Binomial Distribution* - represents probabilities for binomially-distributed
            travel time components (or subcomponents), only for positive delay values
            (negative portions of the distribution are discarded and the distribution is
            rescaled accordingly)
        *   *Multinomial Distribution* - represents probabilities for multinomially-distributed
            travel time components (or subcomponents), only for positive delay values
            (negative portions of the distribution are discarded and the distribution is
            rescaled accordingly). Accepts an arbitrary number of categories for the
            component categorical distributions, which must be provided as a probability
            distribution object (e.g., Arbitrary Distribution, Normal Distribution, Lognormal
            Distribution).
        *   *Poisson Distribution* - Count object for randomly occuring independent events
            Requires average number of events per unit time
        
    Authors:
        teo@sfcta.org, 12/19/2013
"""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
@@@ 1. HOUSEKEEPING
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

from tt_common import *
from cumulative_distribution import CumulativeDistribution


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
@@@ 2. GENERIC TIME AND COUNT DISTRIBUTION CLASSES
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#####################################################
### ARBITRARY DISTRIBUTION
#####################################################

class ArbitraryDistribution:
    """
        Class of duration objects for arbitrarily distruted timeframes
        
        .. note:: This distribution can capture negative values of delay (i.e. reductions in delay).

        .. note::
           The sum of all probabilities should be 1 (but will be rescaled if they are approximate).
           Probabilities of no increase and no reduction will both be counted and should not be provided redundantly.
        
        Requires list of probabilities (Decimal, float OK) for each duration in seconds as follows:
        
        :param increase_prob: ``[prob_0_sec_increase, prob_1_sec_increase, prob_2_sec_increase, ...]``
        :param reduction_prob: ``[prob_0_sec_reduction, prob_1_sec_reduction, prob_2_sec_reduction, ...]``
        
    """
    def __init__(self, increase_prob=[], reduction_prob=[]):
        if(VERBOSITY > 3):
            print time.ctime() + ': ArbitraryDistribution: creating arbitrary distribution object,'
            try:
                print time.ctime() + ': ArbitraryDistribution: probability of no increase in time: ' + str(increase_prob[0]) + '; max increase: ' + str(-1 + len(increase_prob))
            except IndexError:
                print time.ctime() + ': ArbitraryDistribution: no increases in time'
            try:
                print time.ctime() + ': ArbitraryDistribution: probability of no reduction in time: ' + str(reduction_prob[0]) + ';max reduction ' + str(-1 + len(reduction_prob))
            except IndexError:
                print time.ctime() + ': ArbitraryDistribution: no reductions in time'
        self.probability_positive = increase_prob
        self.probability_negative = reduction_prob

        # calculate cumulative probability
        cum_prob = Decimal(0)
        for prob in self.probability_positive:
            assert_numeric(prob)
            cum_prob += Decimal(prob)
        for prob in self.probability_negative:
            assert_numeric(prob)
            cum_prob += Decimal(prob)

        if(cum_prob == 0):
            raise AssertionError('Arbitrary distribution object seeded with zero probability.')

        # rescale based on cumulative probability
        if(VERBOSITY > 4):
            print time.ctime() + ': ArbitraryDistribution: rescaling probabilities, base cumulative probability is ' + str(cum_prob) + ' (should be near 1)'
        scaled_probability_positive = []
        scaled_probability_negative = []
        for prob in self.probability_positive:
            scaled_probability_positive.append(Decimal(prob)/cum_prob)
        self.probability_positive = scaled_probability_positive
        for prob in self.probability_negative:
            scaled_probability_negative.append(Decimal(prob)/cum_prob)
        self.probability_negative = scaled_probability_negative
        
    def prob(self, time_sec):
        """ Given an integer number of seconds of delay, returns the Decimal probability
            of that many seconds of delay as modeled by this probability distribution.
        """
        prob_this_time_sec = Decimal(0)
        if (time_sec >= 0):
            try:
                prob_this_time_sec += self.probability_positive[time_sec]
            except IndexError:
                pass
        if (time_sec <= 0):
            try:
                prob_this_time_sec += self.probability_negative[-1*time_sec]
            except IndexError:
                pass
        return prob_this_time_sec

    def max_delay(self):
        """ Returns the maximum possible delay, in seconds, as modeled by this probability
            distribution.
        """
        return max(0, -1 + len(self.probability_positive))

    def min_delay(self):
        """ Returns the greatest possible reduction in delay, in negative seconds, as modeled
            by this probability distribution.
        """
        return min(0, -1 * (-1 + len(self.probability_negative)))

 
#####################################################
### NORMAL DISTRIBUTION
#####################################################


class NormalDistribution:
    """ Class of duration objects for normally-distributed headways and travel time
    
        Requires mean and standard deviation of time, and maximum possible time, in seconds
        
        OPTIONAL: probability of delay or that distribution does apply (assumed to be 1 if not provided)

        :param mu: mean time
        :param sigma: standard deviation of time
        :param max_time: maximum possible time to be modeled
        :param prob_appl: probability that the delay distribution applies
        :type prob_appl: optional
        
    """
    def __init__(self, mu, sigma, max_time, prob_appl=1):
        if(VERBOSITY > 3):
            print time.ctime() + ': NormalDistribution: creating normal distribution object, average delay: ' + str(mu) + '; standard deviation: ' + str(sigma)
        assert_numeric(mu)
        assert_numeric(sigma)
        assert_numeric(max_time)
        assert_numeric(prob_appl)
        self.mean = Decimal(mu)
        self.variance = Decimal(sigma)**2
        self.probability = []
        self.max = max_time
        self.prob_appl = Decimal(prob_appl)
        self.prob_notappl = 1 - self.prob_appl
        if(self.prob_notappl > 1 or self.prob_notappl < 0):
            raise AssertionError('ERROR: Normal distribution called with probability of applicability of ' + str(self.prob_appl))

        # zero-variance has a singular distrtibution
        if(self.variance == 0):
            self.probability.append(self.prob_notappl)
            scalar = int(self.mean)
            while True:
                try:
                    self.probability[scalar] = self.prob_appl
                    break
                except IndexError:
                    self.probability.append(0)
            return

        # normal distribution
        duration_sec = 0
        cum_prob = Decimal(0)
        while(duration_sec <= self.max and cum_prob < APPROXIMATE_CERTAINTY):
            this_prob = Decimal( math.exp(-1 * ((duration_sec-self.mean) ** 2) / (2*self.variance)) / math.sqrt(2 * Decimal(math.pi) * self.variance) )
            # scale to probability of the distribution applying
            this_prob *= self.prob_appl
            # if zero duration, add in probability of non-applicability
            if(duration_sec == 0):
                this_prob += self.prob_notappl
            self.probability.append(this_prob)
            cum_prob += this_prob
            if(VERBOSITY > 9):
                print time.ctime() + ': NormalDistribution: assigned normal distribution: ' + str(duration_sec) + ', probability: ' + str(this_prob)
                print time.ctime() + ': NormalDistribution: cumulative probability now: ' + str(cum_prob)
            duration_sec += 1

        # rescale based on cumulative probability
        if(VERBOSITY > 4):
            print time.ctime() + ': NormalDistribution: rescaling probabilities, base cumulative probability is ' + str(cum_prob) + ' (should be near 1)'
        scaled_probability = []
        for prob in self.probability:
            scaled_probability.append(prob/cum_prob)
        self.probability = scaled_probability


    def prob(self, duration_sec):
        """ Given an integer number of seconds of delay, returns the Decimal probability
            of that many seconds of delay as modeled by this probability distribution.
        """
        try:
            return self.probability[duration_sec]
        except IndexError:
            return 0

    def max_delay(self):
        """ Returns the maximum possible delay, in seconds, as modeled by this probability
            distribution.
        """
        return -1 + len(self.probability)

    def min_delay(self):
        """ Returns the minimum possible delay, in seconds, as modeled by this probability
            distribution. (0 seconds for this distribution as written.)
        """
        return 0


#####################################################
### LOGNORMAL DISTRIBUTION
#####################################################

class LognormalDistribution:
    """ Class of duration objects for lognormally-distributed timeframes
    
        Requires EITHER mu, sigma, max log-time in log-seconds for normally distributed variable of which the timeframe is the log
        OR mean and standard deviation of time, and maximum possible time, in seconds (use ``from_mean_stdev`` class method)
        
        OPTIONAL: probability of delay or that distribution does apply (assumed to be 1 if not provided)

        :param mu: mean time for normally-distributed variable
        :param sigma: standard deviation of time for normally-distributed variable
        :param max_time: maximum possible time to be modeled
        :param prob_appl: probability that the delay distribution applies
        :type prob_appl: optional
        
    """
    def __init__(self, mu, sigma, max_log_time, prob_appl=1):
        if(VERBOSITY > 3):
            print time.ctime() + ': LognormalDistribution: creating lognormal distribution, mu=' + str(mu) + ', sigma=' + str(sigma)
        assert_numeric(mu)
        assert_numeric(sigma)
        assert_numeric(max_log_time)
        assert_numeric(prob_appl)
        self.mu = Decimal(mu)
        self.sigma = Decimal(sigma)
        self.mean = Decimal(math.exp(self.mu + (self.sigma ** 2)/2))
        self.sd = Decimal(math.sqrt( (math.exp(self.sigma **2) - 1) * math.exp(2 * self.mu + self.sigma ** 2) ) )
        self.max = Decimal(math.exp(max_log_time))
        self.prob_appl = Decimal(prob_appl)
        self.prob_notappl = 1 - self.prob_appl
        if(self.prob_notappl > 1 or self.prob_notappl < 0):
            raise AssertionError('ERROR: Lognormal distribution called with probability of applicability of ' + str(self.prob_appl))
        self.probability = [self.prob_notappl] # impossible to have 0 duration with lognormal distribution unless there is no delay

        # zero-variance has a singular distrtibution
        if self.sigma == 0:
            scalar = int(self.mean)
            while True:
                try:
                    self.probability[scalar] = self.prob_appl
                    break
                except IndexError:
                    self.probability.append(Decimal(0))
    
        # lognormal distribution
        duration_sec = 1
        cum_prob = self.prob_notappl
        while(duration_sec <= self.max and cum_prob < APPROXIMATE_CERTAINTY):
            this_prob = 1/(duration_sec * Decimal(math.sqrt(2 * math.pi)) * self.sigma) * Decimal(math.exp(-1 * (Decimal(math.log(duration_sec))-self.mu)**2 / (2 * self.sigma**2)))
            # scale to probability of the distribution applying
            this_prob *= self.prob_appl
            self.probability.append(this_prob)
            cum_prob += this_prob
            if(VERBOSITY > 9):
                print time.ctime() + ': LognormalDistribution: assigned lognormal distribution: ' + str(duration_sec) + ', probability: ' + str(this_prob)
                print time.ctime() + ': LognormalDistribution: cumulative probability now: ' + str(cum_prob)
            duration_sec += 1


        # rescale based on cumulative probability
        if(VERBOSITY > 4):
            print time.ctime() + ': LognormalDistribution: rescaling probabilities, base cumulative probability is ' + str(cum_prob) + ' (should be near 1)'
        scaled_probability = []
        for prob in self.probability:
            scaled_probability.append(prob/cum_prob)
        self.probability = scaled_probability

    @classmethod
    def from_mean_stdev(cls, mean, stdev, max_time, prob_appl=1):
        """ Alternate constructor for LognormalDistribution

            Requires mean and standard deviation of time, and maximum possible time, in seconds
        
            OPTIONAL: probability of delay or that distribution does apply (assumed to be 1 if not provided)

            :param mean: mean time for lognormally-distributed variable
            :param stdev: standard deviation of time for lognormally-distributed variable
            :param max_time: maximum possible time to be modeled
            :param prob_appl: probability that the delay distribution applies
            :type prob_appl: optional
            
        """
        if(VERBOSITY > 3):
            print time.ctime() + ": LognormalDistribution: creating lognormal distribution,  mean=" + str(mean) + ", stdev=" + str(stdev)
        assert_numeric(mean)
        assert_numeric(stdev)
        assert_numeric(max_time)
        assert_numeric(prob_appl)
        mean = Decimal(mean)
        sd = Decimal(stdev)
        # zero-variance has a singular distrtibution
        if sd == 0:
            return NormalDistribution(mean, 0, max_time, prob_appl)
        mu = Decimal(math.log(mean**2 / Decimal(math.sqrt(sd**2 + mean**2))))
        sigma = Decimal(math.sqrt(Decimal(math.log(1 + (sd**2)/(mean**2)))))
        return cls(mu, sigma, math.log(max_time), prob_appl)


    def prob(self, duration_sec):
        """ Given an integer number of seconds of delay, returns the Decimal probability
            of that many seconds of delay as modeled by this probability distribution.
        """
        try:
            return self.probability[duration_sec]
        except IndexError:
            return 0

    def max_delay(self):
        """ Returns the maximum possible delay, in seconds, as modeled by this probability
            distribution.
        """
        return -1 + len(self.probability)

    def min_delay(self):
        """ Returns the minimum possible delay, in seconds, as modeled by this probability
            distribution. (0 seconds for this distribution as written.)
        """
        return 0            


#####################################################
### BINOMIAL DISTRIBUTION
#####################################################


class BinomialDistribution:
    """ Class of duration objects for binomially-distributed delays
    
        Requires number of trials (n), probability of encounter (p),
        delay per encounter (fixed, seconds)

        :param binomial_n: Number of trials
        :param binomial_p: Probability of encounter
        :param delay_per_encounter: Fixed delay per encounter in seconds
    """
    def __init__(self, binomial_n, binomial_p, delay_per_encounter):
        if(VERBOSITY > 3):
            print time.ctime() + ': BinomialDistribution: creating binomial distribution, n=' + str(binomial_n)
        assert_numeric(binomial_n)
        assert_numeric(binomial_p)
        assert_numeric(delay_per_encounter)
        self.n = int(binomial_n)
        self.p = Decimal(binomial_p)
        self.delay_sec = int(delay_per_encounter)

        # binomial distribution
        self.probability = []   # probability of delay (sec)
        self.probability_encounters = []    # probability of k encounters
        k = 0   # number of encounters (successes)
        while(k <= self.n):
            nck = Decimal(math.factorial(self.n))/(math.factorial(k)*math.factorial(self.n - k)) # nCk, combinations of size k from n elements
            binomial_pmf = nck * (self.p ** k) * ((1 - self.p) ** (self.n - k))
            self.probability_encounters.append(binomial_pmf)
            while(len(self.probability) <= k * self.delay_sec):
                self.probability.append(0)
            self.probability[k * self.delay_sec] = binomial_pmf
            k += 1

    def prob(self, duration_sec):
        """ Given an integer number of seconds of delay, returns the Decimal probability
            of that many seconds of delay as modeled by this probability distribution.
        """
        try:
            return self.probability[duration_sec]
        except IndexError:
            return 0

    def prob_count(self, num_instances):
        """ Given an integer number of encounters of delay, returns the Decimal probability
            of that many encounters of delay as modeled by this probability distribution.
        """
        try:
            return self.probability_encounters[num_instances]
        except IndexError:
            return 0
        
    def max_delay(self):
        """ Returns the maximum possible delay, in seconds, as modeled by this probability
            distribution.
        """
        return -1 + len(self.probability)

    def min_delay(self):
        """ Returns the minimum possible delay, in seconds, as modeled by this probability
            distribution. (0 seconds for this distribution as written.)
        """
        return 0


#####################################################
### MULTINOMIAL DISTRIBUTION
#####################################################

class MultinomialDistribution:
    """ Class of duration objects for multinomially-distributed delays
    
        Requires number of trials (n), delay per trial (probability distribution)

        :param multinomial_n: Number of trials
        :param delay_per_trial: delay object which must have a prob() function
            that returns the probability of delay, in seconds, as passed to the function,
            a max_delay() function which returns the maximum possible delay, in seconds,
            and a min_delay() function which returns the minimum possible delay (0 or
            negative), in seconds

    """
    def __init__(self, multinomial_n, delay_per_trial):
        if(VERBOSITY > 3):
            print time.ctime() + ': MultinomialDistribution: creating multinomial distribution, n=' + str(multinomial_n)
        assert_numeric(multinomial_n)
        self.n = int(multinomial_n)
        self.categorical_dist = delay_per_trial

        # multinomial distribution
        self.trials = []
        for k in range(multinomial_n):
            self.trials.append(self.categorical_dist)
        self.distribution = CumulativeDistribution(self.trials)

        # extract probabilities
        self.probability = []
        delay_sec = 0
        max_delay = self.distribution.max_delay()
        while(delay_sec <= max_delay):
            self.probability.append(self.distribution.prob(delay_sec))
            delay_sec += 1
        

    def prob(self, duration_sec):
        """ Given an integer number of seconds of delay, returns the Decimal probability
            of that many seconds of delay as modeled by this probability distribution.
        """
        try:
            return self.probability[duration_sec]
        except IndexError:
            return 0
        
    def max_delay(self):
        """ Returns the maximum possible delay, in seconds, as modeled by this probability
            distribution.
        """
        return -1 + len(self.probability)

    def min_delay(self):
        """ Returns the minimum possible delay, in seconds, as modeled by this probability
            distribution. (0 seconds for this distribution as written.)
        """
        return 0


#####################################################
### POISSON DISTRIBUTION
#####################################################

class PoissonDistribution:
    """ Class of count objects for randomly occuring independent events
    
        Requires average number of events per unit time
        
        :param p_lambda: average number of events per unit time
    """
    def __init__(self, p_lambda):
        if(VERBOSITY > 3):
            print time.ctime() + ": PoissonDistribution creating poisson distribution, lambda=" + str(p_lambda)
        assert_numeric(p_lambda, True)
        self.p_lambda = Decimal(p_lambda)
        if self.p_lambda == 0:
            self.p_lambda = int(self.p_lambda) # Decimal work-around; cannot take Decimal(0)**0
        self.probability = [] # array where the value of each index is the probability of [index] events occuring in one unit time
        
        # calculate poisson pmf
        max_count = int(math.ceil(self.p_lambda + MAX_DEVIATIONS * self.p_lambda))
        cum_prob = Decimal(0)
        k = 0

        while(cum_prob < APPROXIMATE_CERTAINTY and k <= max_count):
            self.probability.append( ((self.p_lambda ** k) / math.factorial(k)) * Decimal(math.exp(-1 * self.p_lambda)) )
            cum_prob += self.probability[k]
            k += 1
            
        # rescale based on cumulative probability
        if(VERBOSITY > 4):
            print time.ctime() + ': PoissionDistribution: rescaling probabilities, base cumulative probability is ' + str(cum_prob) + ' (should be near 1)'
        scaled_probability = []
        for prob in self.probability:
            scaled_probability.append(prob/cum_prob)
        self.probability = scaled_probability
        
    def prob(self, count):
        """ Given an integer number of events, returns the Decimal probability
            of that many events as modeled by this probability distribution.
        """
        if (count < 0):
            return 0
        try:
            return self.probability[count]
        except IndexError:
            return 0

    def max_count(self):
        """ Returns the maximum possible number of events, as modeled by this probability
            distribution.
        """
        return -1 + len(self.probability)
    
    def min_count(self):
        """ Returns the minimum possible number of events, as modeled by this probability
            distribution. (0 events for this distribution as written.)
        """
        return 0
