"""
    Delay Distribution for Traffic Signals (dd_traffic_signal)
    ==========================================================
    Module for delay caused by traffic signals

    Authors:
        teo@sfcta.org, 12/16/2013
"""


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
@@@ 1. HOUSEKEEPING
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

from tt_common import *


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
@@@ 2. IMPORT GENERIC TRAVEL TIME DISTRIBUTIONS
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

from cumulative_distribution import CumulativeDistribution

from time_count_distributions_generic \
    import  ArbitraryDistribution, \
            NormalDistribution, \
            LognormalDistribution, \
            BinomialDistribution, \
            MultinomialDistribution, \
            PoissonDistribution


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
@@@ 3. TRAFFIC SIGNAL DELAY DISTRIBUTION CLASS
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class TrafficSignal:
    """ Delay object for traffic signals
    
        Requires cycle time, green time, and any fixed delay beyond waiting for green (all in seconds)

        :param cycle_time: length of one full signal cycle, in seconds
        :param green_time: length of green phase for bus or tram (per cycle), in seconds
        :param fixed_delay: fixed delay incurred beyond waiting for green phase (may apply in certain cases
            where signal is actuated by arrival of bus or tram), in seconds
        :type fixed_delay: optional, defaults to 0

        .. note:: Include total of all green phases that may exist within a cycle in ``green_time``.
        
    """
    def __init__(self, cycle_time, green_time, fixed_delay=Decimal(0)):
        if(VERBOSITY > 3):
            print time.ctime() + ': TrafficSignal: creating signal object: cycle time: ' + str(cycle_time) + '; green time: ' + str(green_time) + '; fixed time: ' + str(fixed_delay)
        assert_decimal(cycle_time)
        assert_decimal(green_time)
        assert_decimal(fixed_delay)
        self.cycle = cycle_time
        self.green = green_time
        self.addl_fixed = fixed_delay
        self.probability = []
        delay_sec = 0
        cum_prob = Decimal(0)
        while delay_sec <= self.cycle:
            if(VERBOSITY > 8):
                print time.ctime() + ': TrafficSignal: cum_prob=' + str(cum_prob)
                print time.ctime() + ': TrafficSignal: delay_sec=' + str(delay_sec)
            if(delay_sec < self.addl_fixed):
                self.probability.append(0)      # impossible to have delay less than fixed
            elif(delay_sec == self.addl_fixed):
                pgreen = self.green/self.cycle  # caveat: doesn't consider progression.  Assumes complete independence among signals, which is probably appropriate given that there is at least one stop between signals in each scenario.
                self.probability.append(pgreen) # probability of minimum delay (arrive at green light)
                cum_prob += pgreen
            else:
                self.probability.append(1/self.cycle)
                cum_prob += 1/self.cycle
            delay_sec += 1
            
        # rescale based on cumulative probability
        if(VERBOSITY > 4):
            print time.ctime() + ': TrafficSignal: rescaling probabilities, base cumulative probability is ' + str(cum_prob) + ' (should be near 1)'
        scaled_probability = []
        for prob in self.probability:
            scaled_probability.append(prob/cum_prob)
        self.probability = scaled_probability

    def prob(self, delay_sec):
        """ Given an integer number of seconds of delay, returns the Decimal probability
            of that many seconds of delay as modeled by this probability distribution.
        """
        try:
            return self.probability[delay_sec]
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

