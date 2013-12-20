"""
    Delay Distribution for Train Stations and Bus Stops (dd_train_station)
    ======================================================================
    Module for delay caused by dwell time at train stations or bus stops

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
@@@ 3. TRAIN STATION DELAY DISTRIBUTION CLASS
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""      

class TrainStation:
    """ Delay object for train stations
    
        Requires fixed delay, hourly boardings, hourly alightings, number of doors,
        boarding time per pax per door, alighting time per pax per door,
        headway object (must have ``prob()`` function that returns the probability of given headway, in seconds, as passed).
        
        Optional: Stop requirement (defaults to ``False``) - means the train stops even if no passengers board or alight

        :param fixed_delay: Fixed delay due to stopping in station or at bus stop (seconds)
        :param doors: Number of boarding and alighting doors on the train or bus
        :param board_demand: Hourly boardings at this station or stop
        :param alight_demand: Hourly alightings at this station or stop
        :param board_pace: Boarding time per passenger per door (may vary based on size of doors or number of steps)
        :param alight_pace: Alighting time per passenger per door (may vary based on size of doors or number of steps)
        :param headway_obj: Time distribution object for train or bus headways. Must have a ``prob()`` function
            that returns the probability of a duration, in seconds, as passed to the function,
            and a ``max_delay()`` function which returns the maximum possible duration, in seconds.
            
            .. note:: typically **headway_obj** will be an instance of a delay object class, although here it is used
                for interstitial time rather than travel time
        :param required_stop: set to ``True`` if the train or bus will stop at this station or stop even if
            no passengers board or alight
        :type required_stop: optional, defaults to ``False``
        
        .. note:: probabilities precise only as specified by APPROXIMATE_CERTAINTY in settings module
        
    """
    def __init__(self, fixed_delay, doors, board_demand, alight_demand, board_pace, alight_pace, headway_obj, required_stop=False):
        if(VERBOSITY > 3):
            print time.ctime() + ': TrainStation: creating train station object, fixed delay: ' + str(fixed_delay) + '; board demand: ' + str(board_demand) + '; board time: ' + str(board_pace) + ' (sec per pax per door)'
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
        """ Internal method for extracting probabilities from ``headway_obj`` """
        if (VERBOSITY > 5):
            print time.ctime() + ': TrainStation: extracting headway probabilities'
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
                    print time.ctime() + ': TrainStation: found headway ' + str(headway_sec) + ' with probability ' + str(this_prob) + '; cumulative headway probability: ' + str(cum_prob)
            headway_sec += 1
        if(VERBOSITY > 6):
            print time.ctime() + ': TrainStation: extracted headway probabilities; cumulative probability: ' + str(cum_prob)
        if(cum_prob > 1):
            raise AssertionError('Coding error: Cumulative headway probabilities exceed 1')
        if(cum_prob < APPROXIMATE_CERTAINTY):
            print time.ctime() + ': TrainStation: WARNING! Cumulative headway probability is only ' + str(cum_prob)
        self.headways_cum_prob = cum_prob
        self.max_headway = max_headway
        self.max_headway_hrs = Decimal(max_headway)/3600

    def calc_pax_distrib(self):
        """ Internal method for calculating probability distribution of boarding and alighting passengers """
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
                print time.ctime() + ': TrainStation: calculating probability for boarding pax. threshold probability: ' + str(APPROXIMATE_CERTAINTY*self.headways_cum_prob)
            max_stdev = Decimal(math.sqrt(self.hourly_board*self.max_headway_hrs))
            max_board_pax = int(math.ceil(max_stdev * MAX_DEVIATIONS))
            while(cum_prob < APPROXIMATE_CERTAINTY and board_pax < max_board_pax):
                if (VERBOSITY > 8):
                    print time.ctime() + ': TrainStation: calculating probability for ' + str(board_pax) + ' boarding passengers; cumulative boarding probability: ' + str(cum_prob)
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
            if(VERBOSITY > 4):
                print time.ctime() + ': TrainStation: rescaling probabilities, base cumulative probability is ' + str(cum_prob) + ' (should be near 1)'
            scaled_probability = []
            for prob in self.board_pax_prob:
                scaled_probability.append(prob/cum_prob)
            self.board_pax_prob = scaled_probability

        if(VERBOSITY > 5):
            print time.ctime() + ': TrainStation: cumulative boarding probability: ' + str(cum_prob)

        # if no passengers
        if(self.hourly_alight == 0):
            self.alight_pax_prob.append(1)
            cum_prob = 1
        else:
            # poisson process for alighting pax
            cum_prob = Decimal(0)
            if (VERBOSITY > 4):
                print time.ctime() + ': TrainStation: calculating probability for alighting pax. threshold probability: ' + str(APPROXIMATE_CERTAINTY*self.headways_cum_prob)
            max_stdev = Decimal(math.sqrt(self.hourly_alight*self.max_headway_hrs))
            max_alight_pax = int(math.ceil(max_stdev * MAX_DEVIATIONS))
            while(cum_prob < APPROXIMATE_CERTAINTY and alight_pax < max_alight_pax):
                if (VERBOSITY > 8):
                    print time.ctime() + ': TrainStation: calculating probability for ' + str(alight_pax) + ' alighting passengers; cumulative alighting probability: ' + str(cum_prob)
                for headway in self.headways:   # [headway_sec, this_prob]
                    headway_hrs = Decimal(headway[0])/3600
                    headway_prob = headway[1]
                    if(VERBOSITY > 8):
                        print time.ctime() + ': TrainStation: calculating alighting probability for:'
                        print time.ctime() + ': TrainStation: headway (hrs), ' + str(headway_hrs) + '; headway probability, ' + str(headway_prob)
                        print time.ctime() + ': TrainStation: hourly alightings, ' + str(self.hourly_alight) + '; alighting passengers, ' + str(alight_pax)
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
            if(VERBOSITY > 4):
                print time.ctime() + ': TrainStation: rescaling probabilities, base cumulative probability is ' + str(cum_prob) + ' (should be near 1)'
            scaled_probability = []
            for prob in self.alight_pax_prob:
                scaled_probability.append(prob/cum_prob)
            self.alight_pax_prob = scaled_probability

        if(VERBOSITY > 5):
            print time.ctime() + ': TrainStation: cumulative alighting probability: ' + str(cum_prob)

    def calculate_delay_probs(self):
        """ Internal method for calculating delay probabilities """
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
                    print time.ctime() + ': TrainStation: set boarding and alighting probability of ' + str(this_prob) + ' for ' + str(total_delay_sec) + ' sec of delay'
                        
        # probabilities all calculated!
        self.probability = delay_probs


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


