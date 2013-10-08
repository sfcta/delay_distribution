""" Script to calculate cumulative distribution of delay given various elements of delay
    teo@sfcta.org, 8/26/2013
"""

import math, decimal, time
from decimal import Decimal

VERBOSITY = 9                               # higher values will produce more feedback (0-10)
decimal.getcontext().prec = 6               # decimal precision
MAX_DEVIATIONS = 5                          # in a normal or lognormal distribution, maximum deviations from the mean that will be analyzed.
                                            # in a poisson distribution, maximum variances from the mean (multiple of the mean) that will be analyzed.
                                            # note: in normal: 3 deviations include 0.997 of values; 4 include 0.99994; 5 include 0.9999994; 6 include 0.999999998
APPROXIMATE_ZERO = Decimal(0.1)**(decimal.getcontext().prec)
APPROXIMATE_CERTAINTY = 1-APPROXIMATE_ZERO
                                            

USAGE = """

Delay Distribution Generator: Calculates cumulative distribution given
various elements of delay. Accurate to 1-second resolution.

USAGE:

python generate_delay_distribution.py [scenario_name] spec_dir

scenario_name:  name of scenario, specifies input and output nomenclature
spec_dir:       path to directory containing csv files with delay information

input file info:
                * one or more files may be provided
                * all files should contain a header row, with data beginning
                  on the second row
                * files are assumed to be formatted in CSV as saved by Excel

input files:
                BaseTravelTime_[scenario_name].csv
                    - columns: Name, Average travel time, Standard deviation
                    - values in seconds
                TravelTimeAdjustments_[scenario_name].csv
                    - columns: delay_sec, probability
                BinomialDistributions_[scenario_name].csv
                    - columns: Name, n (number of trials), p (probability of
                      encounter at each trial), delay per encounter (seconds)
                TurningVehicles_[scenario_name].csv
                    - columns: Turning Point Set, Number of Turning Points,
                      delay_sec (per point), probability (per point)
                    - Turning Point Set is a string (e.g.: 'A') that refers to
                      a set of turning points with identical delay distributions
                      NOTE that all data for a given set must be grouped
                      consecutively (cannot give data for A, then B, then A)
                    - Number of Turning Points is the number of turning points
                      in the set
                    - delay_sec is a possible delay, in seconds, for turning
                      points in the present set
                    - probability is the probibility of delay_sec delay for
                      turning points in the present set
                TrafficSignals_[scenario_name].csv
                    - columns: Name, Cycle time, Green time, Fixed delay
                    - values in seconds
                StopSigns_[scenario_name].csv [**NOT YET BUILT OUT**]
                    - columns: Name, Fixed delay, Wait probability, Max wait
                    - values in seconds (except probability)
                PedXings_[scenario_name].csv [**NOT YET BUILT OUT**]
                    - columns: Name, Delay probability, Max delay (seconds)
                StationsStops_[scenario_name].csv
                    - columns: Name, Fixed delay (seconds), Number of doors,
                               Hourly boardings, Hourly alightings,
                               Boarding time per passenger per door (seconds),
                               Alighting time per passenger per door (seconds),
                               Mean headway (minutes),
                               Standard deviation of headway (minutes),
                               Stop requirement
                                   (1 or TRUE or YES if stop is required;
                                    0 or FALSE or NO if stop can be skipped
                                    when no pax wish to board or alight)
"""

def assert_numeric(obj):
    if not isinstance(obj, (int, long, float, Decimal)):
        raise AssertionError('Data error: ' + str(obj) + ' is not a number')

def assert_decimal(obj):
    if not isinstance(obj, Decimal):
        if not isinstance(obj, (int, long, float)):
            raise AssertionError('Data error: ' + str(obj) + ' is not a number')
        raise AssertionError('Coding error: ' + str(obj) + ' has not been converted to Decimal')

class ArbitraryDistribution:
    """ Duration object for arbitrarily distruted timeframes
        Requires list of probabilities for each duration, in seconds as follows:
        increase_prob = [prob_0_sec_increase, prob_1_sec_increase, prob_2_sec_increase, ...]
        reduction_prob = [prob_0_sec_reduction, prob_1_sec_reduction, prob_2_sec_reduction, ...]
        probabilities of no increase and no reduction will be added separately
        NOTE: This distribution can capture negative values of delay (i.e. reductions in delay)
    """
    def __init__(self, increase_prob=[], reduction_prob=[]):
        if(VERBOSITY > 3):
            print time.ctime()
            print 'creating arbitrary distribution object,'
            try:
                print 'probability of no increase in time: ' + str(increase_prob[0])
                print 'max increase: ' + str(-1 + len(increase_prob))
            except IndexError:
                print 'no increases in time'
            try:
                print 'probability of no reduction in time: ' + str(reduction_prob[0])
                print 'max reduction ' + str(-1 + len(reduction_prob))
            except IndexError:
                print 'no reductions in time'
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
        scaled_probability_positive = []
        scaled_probability_negative = []
        for prob in self.probability_positive:
            scaled_probability_positive.append(Decimal(prob)/cum_prob)
        self.probability_positive = scaled_probability_positive
        for prob in self.probability_negative:
            scaled_probability_negative.append(Decimal(prob)/cum_prob)
        self.probability_negative = scaled_probability_negative
        
    def prob(self, time_sec):
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
        return max(0, -1 + len(self.probability_positive))

    def min_delay(self):
        return min(0, -1 * (-1 + len(self.probability_negative)))
        

class NormalDistribution:
    """ Duration object for normally-distributed headways and travel time
        Requires mean and standard deviation of time, and maximum possible time, in seconds
        OPTIONAL: probability of delay or that distribution does apply (assumed to be 1 if not provided)
    """
    def __init__(self, mu, sigma, max_time, prob_appl=1):
        if(VERBOSITY > 3):
            print time.ctime()
            print 'creating normal distribution object,'
            print 'average delay: ' + str(mu)
            print 'standard deviation: ' + str(sigma)
        assert_numeric(mu)
        assert_numeric(sigma)
        assert_numeric(max_time)
        assert_numberic(prob_appl)
        self.mean = Decimal(mu)
        self.variance = Decimal(sigma)**2
        self.probability = []
        self.max = max_time
        self.prob_appl = Decimal(prob_appl)
        self.prob_notappl = 1 - self.prob_appl
        if(self.prob_notappl > 1 or self.prob_notappl < 0):
            raise AssertionError('ERROR: Normal distribution called with probability of applicability of ' + str(self.prob_appl))

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
                print 'assigned normal distribution: ' + str(duration_sec) + ', probability: ' + str(this_prob)
                print 'cumulative probability now: ' + str(cum_prob)
            duration_sec += 1

        # rescale based on cumulative probability
        scaled_probability = []
        for prob in self.probability:
            scaled_probability.append(prob/cum_prob)
        self.probability = scaled_probability


    def prob(self, duration_sec):
        try:
            return self.probability[duration_sec]
        except IndexError:
            return 0

    def max_delay(self):
        return -1 + len(self.probability)

    def min_delay(self):
        return 0



class LognormalDistribution:
    """ Duration object for lognormally-distributed timeframes
        Requires mean and standard deviation of time, and maximum possible time, in seconds.
        OPTIONAL: probability of delay or that distribution does apply (assumed to be 1 if not provided)
    """
    def __init__(self, mean, stdev, max_time, prob_appl=1):
        if(VERBOSITY > 3):
            print time.ctime() + ": creating lognormal distribution"
            print time.ctime() + ": mean=" + str(mean) + ", stdev=" + str(stdev)
        assert_numeric(mean)
        assert_numeric(stdev)
        assert_numeric(max_time)
        assert_numeric(prob_appl)
        self.mean = Decimal(mean)
        self.sd = Decimal(stdev)
        self.mu = Decimal(math.log(self.mean**2 / Decimal(math.sqrt(self.sd**2 + self.mean**2))))
        self.sigma = Decimal(math.sqrt(Decimal(math.log(1 + (self.sd**2)/(self.mean**2)))))
        self.max = max_time
        self.prob_appl = Decimal(prob_appl)
        self.prob_notappl = 1 - self.prob_appl
        if(self.prob_notappl > 1 or self.prob_notappl < 0):
            raise AssertionError('ERROR: Lognormal distribution called with probability of applicability of ' + str(self.prob_appl))
        self.probability = [self.prob_notappl] # impossible to have 0 duration with lognormal distribution unless there is no delay
    
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
                print 'assigned lognormal distribution: ' + str(duration_sec) + ', probability: ' + str(this_prob)
                print 'cumulative probability now: ' + str(cum_prob)
            duration_sec += 1

        # rescale based on cumulative probability
        scaled_probability = []
        for prob in self.probability:
            scaled_probability.append(prob/cum_prob)
        self.probability = scaled_probability


    def prob(self, duration_sec):
        try:
            return self.probability[duration_sec]
        except IndexError:
            return 0

    def max_delay(self):
        return -1 + len(self.probability)

    def min_delay(self):
        return 0            
        

class BinomialDistribution:
    """ Duration object for binomially-distributed delays
        Requires number of trials (n), probability of encounter (p),
        delay per encounter (fixed, seconds)
    """
    def __init__(self, binomial_n, binomial_p, delay_per_encounter):
        if(VERBOSITY > 3):
            print time.ctime() + ': creating binomial distribution, n=' + str(binomial_n)
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
        try:
            return self.probability[duration_sec]
        except IndexError:
            return 0

    def prob_count(self, num_instances):
        try:
            return self.probability_encounters[num_instances]
        except IndexError:
            return 0
        
    def max_delay(self):
        return -1 + len(self.probability)

    def min_delay(self):
        return 0


class MultinomialDistribution:
    """ Duration object for multinomially-distributed delays
        Requires number of trials (n), delay per trial (probability distribution)
    """
    def __init__(self, multinomial_n, delay_per_trial):
        if(VERBOSITY > 3):
            print time.ctime() + ': creating multinomial distribution, n=' + str(multinomial_n)
        assert_numeric(multinomial_n)
        self.n = int(multinomial_n)
        self.categorical_dist = delay_per_trial
        self.prob_encounter = 1-Decimal(prob_encounter)

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
        try:
            return self.probability[duration_sec]
        except IndexError:
            return 0

    def prob_count(self, num_instances):
        try:
            return self.probability_encounters[num_instances]
        except IndexError:
            return 0
        
    def max_delay(self):
        return -1 + len(self.probability)

    def min_delay(self):
        return 0


class TrafficSignal:
    """ Delay object for traffic signals
        Requires cycle time, green time, and any fixed delay beyond waiting for green (all in seconds)
    """
    def __init__(self, cycle_time, green_time, fixed_delay=Decimal(0)):
        if(VERBOSITY > 3):
            print time.ctime()
            print 'creating signal object,'
            print 'cycle time: ' + str(cycle_time)
            print 'green time: ' + str(green_time)
            print 'fixed time: ' + str(fixed_delay)
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
                print 'cum_prob=' + str(cum_prob)
                print 'delay_sec=' + str(delay_sec)
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
        scaled_probability = []
        for prob in self.probability:
            scaled_probability.append(prob/cum_prob)
        self.probability = scaled_probability

    def prob(self, delay_sec):
        try:
            return self.probability[delay_sec]
        except IndexError:
            return 0

    def max_delay(self):
        return -1 + len(self.probability)
    
    def min_delay(self):
        return 0


class TrainStation:
    """ Delay object for train stations
        Requires fixed delay, hourly boardings, hourly alightings, number of doors,
        boarding time per pax per door, alighting time per pax per door,
        headway object (must have prob() function that returns the probability of given headway, in seconds, as passed).
        Optional: Stop requirement (defaults to False) - means the train stops even if no passengers board or alight
        Note: probabilities precise only as specified by APPROXIMATE_CERTAINTY
    """
    def __init__(self, fixed_delay, doors, board_demand, alight_demand, board_pace, alight_pace, headway_obj, required_stop=False):
        if(VERBOSITY > 3):
            print time.ctime()
            print 'creating train station object,'
            print 'fixed delay: ' + str(fixed_delay)
            print 'board demand: ' + str(board_demand)
            print 'board time: ' + str(board_pace) + ' (sec per pax per door)'
        assert_decimal(fixed_delay)
        assert_decimal(board_demand)
        assert_decimal(alight_demand)
        assert_decimal(doors)
        assert_decimal(board_pace)
        assert_decimal(alight_pace)
        self.required = required_stop   # boolean
        self.addl_fixed = fixed_delay
        self.hourly_board = board_demand
        self.hourly_alight = alight_demand
        self.train_doors = doors
        self.board_sec = board_pace
        self.alight_sec = alight_pace
        self.extract_headway_probs(headway_obj)
        self.calc_pax_distrib()         # calculate distribution of boarding and alighting passenger counts
        self.calculate_delay_probs()
        
    def extract_headway_probs(self, headway_obj):
        if (VERBOSITY > 5):
            print time.ctime()
            print 'extracting headway probabilities'
        headway_sec = 0
        max_headway = headway_obj.max_delay()
        cum_prob = Decimal(0)
        self.headways = []
        while(headway_sec <= max_headway):
            this_prob = Decimal(headway_obj.prob(headway_sec))
            if(this_prob > 0):
                self.headways.append([headway_sec, this_prob])
                cum_prob += this_prob
                if(VERBOSITY > 9):
                    print 'found headway ' + str(headway_sec) + ' with probability ' + str(this_prob)
                    print 'cumulative headway probability: ' + str(cum_prob)
            headway_sec += 1
        if(VERBOSITY > 6):
            print 'extracted headway probabilities; cumulative probability: ' + str(cum_prob)
        if(cum_prob > 1):
            raise AssertionError('Coding error: Cumulative headway probabilities exceed 1')
        if(cum_prob < APPROXIMATE_CERTAINTY):
            print 'WARNING! Cumulative headway probability is only ' + str(cum_prob)
        self.headways_cum_prob = cum_prob
        self.max_headway = max_headway
        self.max_headway_hrs = Decimal(max_headway)/3600

    def calc_pax_distrib(self):
        board_pax = 0
        alight_pax = 0
        self.board_pax_prob = []
        self.alight_pax_prob = []

        # if no passengers
        if(self.hourly_board == 0):
            self.board_pax_prob.append(1)
            cum_prob = 1
        else:
            # poisson process for boarding pax
            cum_prob = Decimal(0)
            if (VERBOSITY > 4):
                print time.ctime()
                print 'calculating probability for boarding pax. threshold probability: ' + str(APPROXIMATE_CERTAINTY*self.headways_cum_prob)
            max_stdev = Decimal(math.sqrt(self.hourly_board*self.max_headway_hrs))
            while(cum_prob < APPROXIMATE_CERTAINTY and board_pax < max_stdev * MAX_DEVIATIONS):
                if (VERBOSITY > 8):
                    print 'calculating probability for ' + str(board_pax) + ' boarding passengers'
                    print 'cumulative boarding probability: ' + str(cum_prob)
                for headway in self.headways:   # [headway_sec, this_prob]
                    headway_hrs = Decimal(headway[0])/3600
                    headway_prob = headway[1]
                    try:
                        this_board_prob = headway_prob * Decimal(math.exp(-1*self.hourly_board*headway_hrs)) * ((self.hourly_board*headway_hrs)**board_pax) / Decimal(math.factorial(board_pax))
                    except decimal.InvalidOperation:
                        this_board_prob = 0
                    while(True):
                        try:
                            self.board_pax_prob[board_pax] += this_board_prob
                            break
                        except IndexError:
                            self.board_pax_prob.append(0)
                    cum_prob += this_board_prob
                board_pax += 1
            # rescale based on cumulative probability
            scaled_probability = []
            for prob in self.board_pax_prob:
                scaled_probability.append(prob/cum_prob)
            self.board_pax_prob = scaled_probability

        if(VERBOSITY > 5):
            print 'cumulative boarding probability: ' + str(cum_prob)

        # if no passengers
        if(self.hourly_alight == 0):
            self.alight_pax_prob.append(1)
            cum_prob = 1
        else:
            # poisson process for alighting pax
            cum_prob = Decimal(0)
            if (VERBOSITY > 4):
                print time.ctime()
                print 'calculating probability for alighting pax. threshold probability: ' + str(APPROXIMATE_CERTAINTY*self.headways_cum_prob)
            max_stdev = Decimal(math.sqrt(self.hourly_alight*self.max_headway_hrs))
            while(cum_prob < APPROXIMATE_CERTAINTY and alight_pax < max_stdev * MAX_DEVIATIONS):
                if (VERBOSITY > 8):
                    print 'calculating probability for ' + str(alight_pax) + ' alighting passengers'
                    print 'cumulative alighting probability: ' + str(cum_prob)
                for headway in self.headways:   # [headway_sec, this_prob]
                    headway_hrs = Decimal(headway[0])/3600
                    headway_prob = headway[1]
                    if(VERBOSITY > 8):
                        print 'calculating alighting probability for:'
                        print 'headway (hrs), ' + str(headway_hrs) + '; headway probability, ' + str(headway_prob)
                        print 'hourly alightings, ' + str(self.hourly_alight) + '; alighting passengers, ' + str(alight_pax)
                    try:
                        this_alight_prob = headway_prob * Decimal(math.exp(-1*self.hourly_alight*headway_hrs)) * ((self.hourly_alight*headway_hrs)**alight_pax) / Decimal(math.factorial(alight_pax))
                    except decimal.InvalidOperation:
                        this_alight_prob = 0
                    while(True):
                        try:
                            self.alight_pax_prob[alight_pax] += this_alight_prob
                            break
                        except IndexError:
                            self.alight_pax_prob.append(0)
                    cum_prob += this_alight_prob
                alight_pax += 1
            # rescale based on cumulative probability
            scaled_probability = []
            for prob in self.alight_pax_prob:
                scaled_probability.append(prob/cum_prob)
            self.alight_pax_prob = scaled_probability

        if(VERBOSITY > 5):
            print 'cumulative alighting probability: ' + str(cum_prob)

    def calculate_delay_probs(self):
        delay_probs = []
        
        # probability of zero delay (no stop)
        if(self.required):
            delay_probs.append(0)   # impossible to have no delay if stop is required
        else:
            delay_probs.append(self.board_pax_prob[0]*self.alight_pax_prob[0]) # no delay if 0 pax board and 0 pax alight
            
        # impossible to have delay less than fixed delay. zero out probabilities from 1 sec delay to fixed delay.
        delay_sec = 1
        while(delay_sec < int(self.addl_fixed)):
            delay_probs.append(0)
            delay_sec += 1

        # fixed delay will occur if train is required to stop and no pax board or alight
        if(self.required):
            delay_probs.append(self.board_pax_prob[0]*self.alight_pax_prob[0])
        else:
            delay_probs.append(0)   # non-required stops will be skipped if no pax board or alight

        # set cumulative delay probabilities for each possible number of boarding or alighting pax
        for board_pax in range(len(self.board_pax_prob)):
            board_delay = board_pace*board_pax/self.train_doors
            board_delay_prob = self.board_pax_prob[board_pax]
            for alight_pax in range(len(self.alight_pax_prob)):
                alight_delay = alight_pace*alight_pax/self.train_doors
                alight_delay_prob = self.alight_pax_prob[alight_pax]
                # we already handled the case where no pax board or alight
                if(board_pax == 0 and alight_pax == 0):
                    continue
                # now assign the delay probability
                total_delay_sec = int(board_delay + alight_delay + self.addl_fixed)
                this_prob = board_delay_prob * alight_delay_prob
                while(True):
                    try:
                        delay_probs[total_delay_sec] += this_prob
                        break
                    except IndexError:
                        delay_probs.append(0)
                if(VERBOSITY > 6):
                    print 'set boarding and alighting probability of ' + str(this_prob) + ' for ' + str(total_delay_sec) + ' sec of delay'
                        
        # probabilities all calculated!
        self.probability = delay_probs


    def prob(self, delay_sec):
        try:
            return self.probability[delay_sec]
        except IndexError:
            return 0

    def max_delay(self):
        return -1 + len(self.probability)
    
    def min_delay(self):
        return 0



class CumulativeDistribution:
    """ Calculates the cumulative distribution of delay, given components of delay
        Requires a set of one or more objects which must each have a prob() function
        that returns the probability of delay, in seconds, as passed to the function,
        a max_delay() function which returns the maximum possible delay, in seconds,
        and a min_delay() function which returns the minimum possible delay (0 or
        negative), in seconds
    """
    def __init__(self, delay_obj_list):
        if(VERBOSITY > 3):
            print time.ctime() + ': creating cumulative distribution object'
        self.probability = []
        self.partial_probs = []
        for trial in range(len(delay_obj_list)):
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
        scaled_probability = []
        if(VERBOSITY > 4):
            print time.ctime() + ': scaling probabilities for cumulative distribution. unscaled total probability is ' + str(self.cum_prob)
        for prob in self.probability:
            scaled_probability.append(prob/self.cum_prob)
        self.probability = scaled_probability

    def calc_cum_probs(self, existing_probs=[1]):
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
        try:
            return self.probability[delay_sec]
        except IndexError:
            return 0
        
    def max_delay(self):
        return -1 + len(self.probability)

    def min_delay(self):
        return 0



if __name__ == "__main__":
    import os, sys, csv, copy, getopt
    
    optlist,args = getopt.getopt(sys.argv[1:],'')
    if len(args) not in [1,2]:
        print USAGE
        sys.exit(2)
    if len(args) == 2:
        SCENARIO    = args[0]
        SPEC_DIR    = args[1]
    else:
        SCENARIO    = ''
        SPEC_DIR    = args[0]
        
    dir_path = os.path.abspath(SPEC_DIR)
    BaseTT_path         = os.path.join(dir_path, 'BaseTravelTime_' + SCENARIO + '.csv')
    TTAdjust_basepath   = os.path.join(dir_path, 'TravelTimeAdjustments_' + SCENARIO)
    TTAdjust_endpath    = '.csv'
    BinomialDist_path   = os.path.join(dir_path, 'BinomialDistributions_' + SCENARIO + '.csv')
    MNDist_path         = os.path.join(dir_path, 'MultinomialDistributions_' + SCENARIO + '.csv')
    TurningVehs_path    = os.path.join(dir_path, 'TurningVehicles_' + SCENARIO + '.csv')
    TrafficSignals_path = os.path.join(dir_path, 'TrafficSignals_' + SCENARIO + '.csv')
    StopSigns_path      = os.path.join(dir_path, 'StopSigns_' + SCENARIO + '.csv')
    PedXings_path       = os.path.join(dir_path, 'PedXings_' + SCENARIO + '.csv')
    StationsStops_path  = os.path.join(dir_path, 'StationsStops_' + SCENARIO + '.csv')

    # Initial notification
    if(VERBOSITY > 6):
        print 'BEGIN RUN'
        print time.ctime()
        print "APPROXIMATE_CERTAINTY: " + str(APPROXIMATE_CERTAINTY)
        print "APPROXIMATE_ZERO: " + str(APPROXIMATE_ZERO)

    # Process base travel time info
    incl_BaseTT = True
    try:
        with open(BaseTT_path, 'rb') as BaseTT_csv:
            BaseSegments = []
            tt_reader = csv.reader(BaseTT_csv, dialect='excel')
            header = tt_reader.next() # throw away the header as it doesn't contain data
            for row in tt_reader:
                name        = str(row[0])
                avg_time    = Decimal(row[1])
                stdev_time  = Decimal(row[2])
                max_calculable_tt = avg_time + MAX_DEVIATIONS * stdev_time
                BaseSegments.append(NormalDistribution(avg_time, stdev_time, max_calculable_tt))
                if(VERBOSITY > 1):
                    print time.ctime() + ': Read base travel time: ' + name
        TotalBaseTT = CumulativeDistribution(BaseSegments)
        outfile_path = os.path.join(dir_path, 'BaseTravelTime_' + SCENARIO + '_cumulative.csv')
        with open(outfile_path, 'wb') as outfile:
            tt_writer = csv.writer(outfile, dialect='excel')
            delay_sec = 0
            max_delay_sec = TotalBaseTT.max_delay()
            tt_writer.writerow(['CUMULATIVE BASE TRAVEL TIME (' + SCENARIO + ')'])
            tt_writer.writerow(['Delay, sec', 'Cumulative Base Travel Time Probability, ' + SCENARIO])
            while delay_sec <= max_delay_sec:
                tt_writer.writerow([delay_sec, TotalBaseTT.prob(delay_sec)])
                delay_sec += 1
            if(VERBOSITY > 0):
                print time.ctime() + ': Wrote cumulative base travel time: ' + outfile_path            
    except IOError:
        print 'No base travel time file found: ' + BaseTT_path
        print 'Excluding base travel time and base variation from analysis'
        incl_BaseTT = False
        
    # Process travel time adjustments
    incl_TTAdjust = True
    file_idx = 1
    thisfile_path = ''
    TTAdjustments = []
    try:
        # keep trying files til we try one that doesn't exist
        while(True):
            if(file_idx == 1):
                thisfile_path = TTAdjust_basepath + TTAdjust_endpath
            else:
                thisfile_path = TTAdjust_basepath + '_' + str(file_idx) + TTAdjust_endpath
            with open(thisfile_path, 'rb') as TTAdjust_csv:
                ProbabilityListIncreasedDelay = []
                ProbabilityListReducedDelay = []
                ttadjust_reader = csv.reader(TTAdjust_csv, dialect='excel')
                header = ttadjust_reader.next() # throw away the header as it doesn't contain data
                for row in ttadjust_reader:
                    delay_sec   = int(row[0])
                    probability = Decimal(row[1])
                    if(delay_sec >= 0):
                        try:
                            ProbabilityListIncreasedDelay.append(Decimal(0))
                            ProbabilityListIncreasedDelay[delay_sec] = probability
                        except IndexError:
                            while len(ProbabilityListIncreasedDelay <= delay_sec):
                                ProbabilityListIncreasedDelay.append(Decimal(0))
                            ProbabilityListIncreasedDelay[delay_sec] = probability
                    else:
                        try:
                            ProbabilityListReducedDelay.append(Decimal(0))
                            ProbabilityListReducedDelay[-1*delay_sec] = probability
                        except IndexError:
                            while(len(ProbabilityListReducedDelay) <= -1*delay_sec):
                                ProbabilityListReducedDelay.append(Decimal(0))
                            ProbabilityListReducedDelay[-1*delay_sec] = probability                    
                    if(VERBOSITY > 7):
                        print time.ctime() + ': Will adjust travel time by ' + str(delay_sec) + ' with probability ' + str(probability)
            TTAdjustments.append(ArbitraryDistribution(ProbabilityListIncreasedDelay,ProbabilityListReducedDelay))
            if(VERBOSITY > 0):
                print time.ctime() + ': Calculated travel time adjustments from: ' + thisfile_path
            file_idx += 1
    except IOError:
        if(file_idx == 1):
            print 'No travel time adustment file found: ' + thisfile_path
            print 'Excluding any adjustments from analysis'
            incl_TTAdjust = False
        else:
            print 'File not found: ' + thisfile_path
            print 'Only including travel time adjustments for first ' + str(file_idx - 1) + ' TravelTimeAdjustment files'
            TotalTTAdjustments = CumulativeDistribution(TTAdjustments)

    # Process binomial distribution data
    incl_BinDists = True
    try:
        with open(BinomialDist_path, 'rb') as BinDists_csv:
            BinomialDistributions = []
            BinDists_reader = csv.reader(BinDists_csv, dialect='excel')
            header = BinDists_reader.next() # throw away the header as it doesn't contain data
            for row in BinDists_reader:
                name        = str(row[0])
                n           = int(row[1])
                p           = Decimal(row[2])
                delay_sec   = int(row[3])
                BinomialDistributions.append(BinomialDistribution(n, p, delay_sec))
                if(VERBOSITY > 1):
                    print time.ctime() + ': Read binomial distribution: ' + name
        TotalBinomiallyDistributedDelay = CumulativeDistribution(BinomialDistributions)
        outfile_path = os.path.join(dir_path, 'BinomialDistributions_' + SCENARIO + '_cumulative.csv')
        with open(outfile_path, 'wb') as outfile:
            BinDists_writer = csv.writer(outfile, dialect='excel')
            delay_sec = 0
            max_delay_sec = TotalBinomiallyDistributedDelay.max_delay()
            BinDists_writer.writerow(['CUMULATIVE DELAY DUE TO BINOMIALLY-DISTRIBUTED FACTORS (' + SCENARIO + ')'])
            BinDists_writer.writerow(['Delay, sec', 'Cumulative Binomially-Distributed Delay Probability, ' + SCENARIO])
            while delay_sec <= max_delay_sec:
                BinDists_writer.writerow([delay_sec, TotalBinomiallyDistributedDelay.prob(delay_sec)])
                delay_sec += 1
            if(VERBOSITY > 0):
                print time.ctime() + ': Wrote cumulative binomially-distributed delay: ' + outfile_path
    except IOError:
        print 'No binomial delay file found: ' + BinomialDist_path
        print 'Excluding any binomially-distrubted delays from analysis'
        incl_BinDists = False


    # Process multinomial distribution data
    incl_MNDists = True
    try:
        with open(MNDist_path, 'rb') as MNDists_csv:
            MultinomialDistributions = []
            MNDists_reader = csv.reader(MNDists_csv, dialect='excel')
            header = MNDists_reader.next() # throw away the header as it doesn't contain data
            for row in MNDists_reader:
                name            = str(row[0])
                num_trials      = int(row[1])
                prob_encounter  = Decimal(row[2])
                mean            = Decimal(row[3])
                stdev           = Decimal(row[4])
                dist_type       = str(row[5]).upper()
                max_delay_per_encounter = mean + MAX_DEVIATIONS * stdev
                if(dist_type == 'NORMAL'):
                    delay_per_encounter = NormalDistribution(mean, stdev, max_delay_per_encounter, prob_encounter)
                elif(dist_type == 'LOGNORMAL'):
                    delay_per_encounter = LognormalDistribution(mean, stdev, max_delay_per_encounter, prob_encounter)
                else:
                    raise AssertionError('ERROR: unsupported delay distribution for multinomial: ' + dist_type + ' in file ' + MNDist_path)
                MultinomialDistributions.append(MultinomialDistribution(num_trials, delay_per_encounter))
                if(VERBOSITY > 1):
                    print time.ctime() + ': Read multinomial distribution: ' + name                
        TotalMultinomiallyDistributedDelay = CumulativeDistribution(MultinomialDistributions)
        outfile_path = os.path.join(dir_path, 'MultinomialDistributions_' + SCENARIO + '_cumulative.csv')
        with open(outfile_path, 'wb') as outfile:
            MNDists_writer = csv.writer(outfile, dialect='excel')
            delay_sec = 0
            max_delay_sec = TotalMultinomiallyDistributedDelay.max_delay()
            MNDists_writer.writerow(['CUMULATIVE DELAY DUE TO MULTINOMIALLY-DISTRIBUTED FACTORS (' + SCENARIO + ')'])
            MNDists_writer.writerow(['Delay, sec', 'Cumulative Multinomially-Distributed Delay Probability, ' + SCENARIO])
            while delay_sec <= max_delay_sec:
                MNDists_writer.writerow([delay_sec, TotalMultinomiallyDistributedDelay.prob(delay_sec)])
                delay_sec += 1
            if(VERBOSITY > 0):
                print time.ctime() + ': Wrote cumulative multinomially-distributed delay: ' + outfile_path
    except IOError:
        print 'No multinomial delay file found: ' + MNDist_path
        print 'Excluding any multinomially-distrubted delays from analysis'
        incl_MNDists = False

    # Process turning vehicles info
    incl_TurningVehs = True
    try:
        with open(TurningVehs_path, 'rb') as TurningVehs_csv:
            if(VERBOSITY > 1):
                print time.ctime() + ': reading in turning vehicle info'
            DelayProbabilityList = []
            TurningPointSets = []
            latest_set = ''
            latest_num_pts = -1
            tv_reader = csv.reader(TurningVehs_csv, dialect='excel')
            header = tv_reader.next() # throw away the header as it doesn't contain data
            for row in tv_reader:
                name        = str(row[0])
                num_points  = int(row[1])
                delay_sec   = int(row[2])
                probability = Decimal(row[2])
                if(name == latest_set or latest_num_pts == -1):
                    if(latest_num_pts <> num_points and name == latest_set):
                        raise AssertionError('Error! Mismatched number of turning points (' + str(num_points) + ') for turning set ' + str(name) + ' (expected: ' + str(latest_set) + ')')
                    latest_set = name
                    latest_num_pts = num_points
                    try:
                        DelayProbabilityList.append(Decimal(0))
                        DelayProbabilityList[delay_sec] = probability
                    except IndexError:
                        while len(DelayProbabilityList <= delay_sec):
                            DelayProbabilityList.append(Decimal(0))
                        DelayProbabilityList[delay_sec] = probability
                else:
                    categorical_distribution = ArbitraryDistribution(DelayProbabilityList)
                    TurningPointSets.append(MultinomialDistribution(latest_num_pts, categorical_distribution))
                    if(VERBOSITY > 1):
                        print time.ctime() + ': Read turning point set: ' + latest_set
                    DelayProbabilityList = []
                    latest_set = name
                    latest_num_pts = num_points
                    try:
                        DelayProbabilityList.append(Decimal(0))
                        DelayProbabilityList[delay_sec] = probability
                    except IndexError:
                        while len(DelayProbabilityList <= delay_sec):
                            DelayProbabilityList.append(Decimal(0))
                        DelayProbabilityList[delay_sec] = probability  
            categorical_distribution = ArbitraryDistribution(DelayProbabilityList)
            TurningPointSets.append(MultinomialDistribution(latest_num_pts, categorical_distribution))
            if(VERBOSITY > 1):
                print time.ctime() + ': Read turning point set: ' + latest_set                      
        TotalTurningVehicleDelay = CumulativeDistribution(TurningPointSets)
        if(VERBOSITY > 0):
            print time.ctime() + ': Calculated turning vehicle delays from: ' + TurningVehs_path            
        outfile_path = os.path.join(dir_path, 'TurningVehicles_' + SCENARIO + '_cumulative.csv')
        with open(outfile_path, 'wb') as outfile:
            tv_writer = csv.writer(outfile, dialect='excel')
            delay_sec = 0
            max_delay_sec = TotalTurningVehicleDelay.max_delay()
            tv_writer.writerow(['CUMULATIVE DELAY DUE TO TURNING VEHICLES (' + SCENARIO + ')'])
            tv_writer.writerow(['Delay, sec', 'Cumulative Turning Vehicle Delay Probability, ' + SCENARIO])
            while delay_sec <= max_delay_sec:
                tv_writer.writerow([delay_sec, TotalTurningVehicleDelay.prob(delay_sec)])
                delay_sec += 1
            if(VERBOSITY > 0):
                print time.ctime() + ': Wrote cumulative turning vehicle delay: ' + outfile_path
    except IOError:
        print 'No turning vehicles file found: ' + TurningVehs_path
        print 'Excluding turning vehicle delay from analysis'
        incl_TurningVehs = False


    # Process traffic signal info
    incl_TrafficSignals = True
    try:
        with open(TrafficSignals_path, 'rb') as TrafficSignals_csv:
            TrafficSignals = []
            signal_reader = csv.reader(TrafficSignals_csv, dialect='excel')
            header = signal_reader.next() # throw away the header as it doesn't contain data
            for row in signal_reader:
                name        = str(row[0])
                cycle_time  = Decimal(row[1])
                green_time  = Decimal(row[2])
                fixed_time  = Decimal(row[3])
                TrafficSignals.append(TrafficSignal(cycle_time, green_time, fixed_time))
                if(VERBOSITY > 1):
                    print time.ctime() + ': Read traffic signal: ' + name
        TotalTrafficSignalDelay = CumulativeDistribution(TrafficSignals)
        outfile_path = os.path.join(dir_path, 'TrafficSignals_' + SCENARIO + '_cumulative.csv')
        with open(outfile_path, 'wb') as outfile:
            signal_writer = csv.writer(outfile, dialect='excel')
            delay_sec = 0
            max_delay_sec = TotalTrafficSignalDelay.max_delay()
            signal_writer.writerow(['CUMULATIVE DELAY DUE TO TRAFFIC SIGNALS (' + SCENARIO + ')'])
            signal_writer.writerow(['Delay, sec', 'Cumulative Signal Delay Probability, ' + SCENARIO])
            while delay_sec <= max_delay_sec:
                signal_writer.writerow([delay_sec, TotalTrafficSignalDelay.prob(delay_sec)])
                delay_sec += 1
            if(VERBOSITY > 0):
                print time.ctime() + ': Wrote cumulative traffic signal delay: ' + outfile_path
    except IOError:
        print 'No traffic signal file found: ' + TrafficSignals_path
        print 'Excluding traffic signal delay from analysis'
        incl_TrafficSignals = False

    # Process stop sign info
    incl_StopSigns = False

    # Process ped xing info
    incl_PedXings = False

    # Process transit stations / stops info
    incl_StationsStops = True
    try:
        with open(StationsStops_path, 'rb') as StationsStops_csv:
            TrainStations = []
            stations_reader = csv.reader(StationsStops_csv, dialect='excel')
            header = stations_reader.next() # throw away the header as it doesn't contain data
            for row in stations_reader:
                name        = str(row[0])
                fixed_delay = Decimal(row[1])
                num_doors   = Decimal(row[2])
                hrly_board  = Decimal(row[3])
                hrly_alight = Decimal(row[4])
                board_pace  = Decimal(row[5])
                alight_pace = Decimal(row[6])
                headway_avg = Decimal(row[7])
                headway_sd  = Decimal(row[8])
                stop_reqd   = str(row[9])
                if(stop_reqd.upper() in ('1','TRUE','YES')):
                    stop_reqd = True
                elif(stop_reqd.upper() in ('0','FALSE','NO')):
                    stop_reqd = False
                else:
                    raise AssertionError('Data error: ' + stop_reqd + ' is not a valid stop requirement specification at station ' + name)
                # only calculate headways up to MAX_DEVIATIONS standard deviations from the mean (beyond is assumed to be negligible)
                if(VERBOSITY > 1):
                    print time.ctime() + ': Reading in delay for station / stop: ' + name
                max_calculable_headway = headway_avg + MAX_DEVIATIONS * headway_sd
                headway_obj = NormalDistribution(60*headway_avg, 60*headway_sd, 60*max_calculable_headway)
                TrainStations.append(TrainStation(fixed_delay, num_doors, hrly_board, hrly_alight, board_pace, alight_pace, headway_obj, stop_reqd))
                if(VERBOSITY > 1):
                    print time.ctime() + ': Calculated delay for station / stop: ' + name
        TotalStationDelay = CumulativeDistribution(TrainStations)
        outfile_path = os.path.join(dir_path, 'StationsStops_' + SCENARIO + '_cumulative.csv')
        with open(outfile_path, 'wb') as outfile:
            station_writer = csv.writer(outfile, dialect='excel')
            delay_sec = 0
            max_delay_sec = TotalStationDelay.max_delay()
            station_writer.writerow(['CUMULATIVE DELAY DUE TO BOARDING & ALIGHTING (' + SCENARIO + ')'])
            station_writer.writerow(['Delay, sec', 'Cumulative Station/Stop Delay Probability, ' + SCENARIO])
            while delay_sec <= max_delay_sec:
                station_writer.writerow([delay_sec, TotalStationDelay.prob(delay_sec)])
                delay_sec += 1
            if(VERBOSITY > 0):
                print time.ctime() + ': Wrote cumulative boarding & alighting delay: ' + outfile_path            
    except IOError:
        print 'No transit stations / stops file found: ' + StationsStops_path
        print 'Excluding boarding & alighting delay from analysis'
        incl_StationsStops = False

    # Calculate total distribution of delay
    if(VERBOSITY > 2):
        print 'Calculating total cumulative delay distribution.'
    delay_component_dists = []
    delay_component_str = ''
    if(incl_BaseTT):
        delay_component_dists.append(TotalBaseTT)
        if(len(delay_component_str) > 0):
            delay_component_str += '; '
        delay_component_str += 'Base travel time'
    if(incl_TTAdjust):
        delay_component_dists.append(TotalTTAdjustments)
        if(len(delay_component_str) > 0):
            delay_component_str += '; '
        delay_component_str += 'Travel time adjustments'
    if(incl_BinDists):
        delay_component_dists.append(TotalBinomiallyDistributedDelay)
        if(len(delay_component_str) > 0):
            delay_component_str += '; '
        delay_component_str += 'Binomially-distributed delays'
    if(incl_MNDists):
        delay_component_dists.append(TotalMultinomiallyDistributedDelay)
        if(len(delay_component_str) > 0):
            delay_component_str += '; '
        delay_component_str += 'Multinomially-distributed delays'
    if(incl_TurningVehs):
        delay_component_dists.append(TotalTurningVehicleDelay)
        if(len(delay_component_str) > 0):
            delay_component_str += '; '
        delay_component_str += 'Turning vehicles'
    if(incl_TrafficSignals):
        delay_component_dists.append(TotalTrafficSignalDelay)
        if(len(delay_component_str) > 0):
            delay_component_str += '; '
        delay_component_str += 'Traffic signals'
    if(incl_StopSigns):
        delay_component_dists.append(TotalStopSignDelay)
        if(len(delay_component_str) > 0):
            delay_component_str += '; '
        delay_component_str += 'Stop signs'
    if(incl_PedXings):
        delay_component_dists.append(TotalPedXingDelay)
        if(len(delay_component_str) > 0):
            delay_component_str += '; '
        delay_component_str += 'Pedestrian crossings'
    if(incl_StationsStops):
        delay_component_dists.append(TotalStationDelay)
        if(len(delay_component_str) > 0):
            delay_component_str += '; '
        delay_component_str += 'Boarding and alighting'
        
    CumulativeCorridorDelay = CumulativeDistribution(delay_component_dists)
    outfile_path = os.path.join(dir_path, 'TotalDelay_' + SCENARIO + '_cumulative.csv')
    with open(outfile_path, 'wb') as outfile:
        cumulative_writer = csv.writer(outfile, dialect='excel')
        delay_sec = 0
        max_delay_sec = CumulativeCorridorDelay.max_delay()
        cumulative_writer.writerow(['CUMULATIVE DELAY ALONG CORRIDOR (' + SCENARIO + ') DUE TO ' + delay_component_str.upper()])
        cumulative_writer.writerow(['Delay, sec', 'Cumulative Delay Probability, ' + SCENARIO + ' (' + delay_component_str + ')'])
        while delay_sec <= max_delay_sec:
            cumulative_writer.writerow([delay_sec, CumulativeCorridorDelay.prob(delay_sec)])
            delay_sec += 1
        if(VERBOSITY > 0):
            print time.ctime() + ': Wrote cumulative corridor delay: ' + outfile_path
            
    # Finish up
    if(VERBOSITY > 0):
        print time.ctime() + ': DELAY DISTRIBUTION CALCULATION - COMPLETED SUCCESSFULLY!'
