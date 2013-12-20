"""
    Cumulative Delays (cumulative_distribution)
    ===========================================
    Cumulative distribution module.

    Calculates the cumulative distribution of delay or travel time (i.e., convolution of component distributions)

    .. note:: Cumulative delay and cumulative travel time are not to be confused with cumulative distribution functions. Although the word
        "cumulative" is used here, to imply the modeling of aggregate delay across analysis zones and delay types, the output distributions are
        probability mass functions (PMFs) that approximate probability density functions (PDFs) on the 1-second scale. Output distributions are NOT
        cumulative distributions (CDFs).

        
    Authors:
        teo@sfcta.org, 12/16/2013
"""


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
@@@ 1. HOUSEKEEPING
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

from tt_common import *

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
@@@ 2. CUMULATIVE DISTRIBUTION CLASS
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class CumulativeDistribution:
    """ Calculates the cumulative distribution of delay, given components of delay.
    
        Requires a list of one or more delay objects which must each have a prob() function
        that returns the probability of delay, in seconds, as passed to the function,
        a max_delay() function which returns the maximum possible delay, in seconds,
        and a min_delay() function which returns the minimum possible delay (0 or
        negative), in seconds.
    """
    def __init__(self, delay_obj_list):
        if(VERBOSITY > 3):
            print time.ctime() + ': CumulativeDistribution: creating cumulative distribution object'
        self.probability = []
        self.partial_probs = []
        self.num_delay_objs = len(delay_obj_list)
        if(VERBOSITY > 5):
            print time.ctime() + ': CumulativeDistribution: calculating cumulative delay from ' + str(self.num_delay_objs) + ' component delay objects'
        if self.num_delay_objs == 0:
            self.probability.append(1)
            return
        for trial in range(self.num_delay_objs):
            if(VERBOSITY > 6):
                if(trial <= 5 or (trial < 100 and trial % 10 == 0) or trial % 50 == 0 or VERBOSITY > 8):
                    print time.ctime() + ': CumulativeDistribution: tallying cumulative delay from component ' + str(trial)
            delay_obj = delay_obj_list[trial]
            max_delay = delay_obj.max_delay()
            self.partial_probs.append([[],[]]) # list for positive and negative delay lists for this trial
            delay_sec = delay_obj.min_delay()
            cum_prob = Decimal(0)
            while(delay_sec <= max_delay and cum_prob < APPROXIMATE_CERTAINTY):
                this_prob = delay_obj.prob(delay_sec)
                if(this_prob < 0):
                    raise AssertionError('Probability is negative: ' + str(this_prob))
                if(delay_sec >= 0):
                    self.partial_probs[trial][0].append(this_prob)
                else: # reduction in delay
                    try:
                        self.partial_probs[trial][1][-1*delay_sec] = this_prob
                    except IndexError:
                        while(len(self.partial_probs[trial][1]) <= -1*delay_sec):
                            self.partial_probs[trial][1].append(0)
                        self.partial_probs[trial][1][-1*delay_sec] = this_prob                        
                cum_prob += this_prob
                delay_sec += 1
        self.num_partials = len(self.partial_probs)
        self.calc_cum_probs()
        # rescale based on cumulative probability
        if(VERBOSITY > 4):
            print time.ctime() + ': CumulativeDistribution: scaling probabilities for cumulative distribution. unscaled total probability is ' + str(self.cum_prob)
        scaled_probability = []
        for prob in self.probability:
            scaled_probability.append(prob/self.cum_prob)
        self.probability = scaled_probability

    def calc_cum_probs(self, existing_probs=[1]):
        """ Internal method for recursive calculation of cumulative probabilities """
        if(VERBOSITY > 6):
            remaining_components = len(self.partial_probs)
            component_id = self.num_delay_objs - remaining_components
            if(component_id <= 5 or (component_id < 100 and component_id % 10 == 0) or component_id % 50 == 0 or VERBOSITY > 8):
                print time.ctime() + ': CumulativeDistribution: calculating cumulative probabilities; remaining components: ' + str(remaining_components)
        new_probs = []
        try:
            addl_probs = self.partial_probs.pop(0)
            incr_probs = addl_probs[0]
            decr_probs = addl_probs[1]
            self.cum_prob = Decimal(0)
        except IndexError:
            self.probability = existing_probs
            return
        # permute by each second of delay in existing distribution
        for delay_sec in range(len(existing_probs)):
            # probabilistically apply incremental delay from additional probabilities
            for incr_delay in range(len(incr_probs)):
                this_delay = delay_sec + incr_delay
                this_prob = existing_probs[delay_sec] * incr_probs[incr_delay]
                self.cum_prob += this_prob
                while(True):
                    try:
                        new_probs[this_delay] += this_prob
                        break
                    except IndexError:
                        new_probs.append(0)
            # probabilistically apply delay reduction from additional probabilities
            for decr_delay in range(len(decr_probs)):
                this_delay = max(0, delay_sec - decr_delay)
                this_prob = existing_probs[delay_sec] * decr_probs[decr_delay]
                self.cum_prob += this_prob
                while(True):
                    try:
                        new_probs[this_delay] += this_prob
                        break
                    except IndexError:
                        new_probs.append(0) 
        self.calc_cum_probs(new_probs)
        
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



