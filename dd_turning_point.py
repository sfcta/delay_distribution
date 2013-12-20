"""
    Delay Distribution for Turning Vehicles (dd_turning_point)
    ==========================================================
    Module for delay caused by right-turning vehicles

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
@@@ 3. TURNING POINT DELAY DISTRIBUTION CLASS
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class TurningPoint:
    """ Class for delay objects for analytically-specified turning points

        .. note:: Models delay due to "pedestrian friction", i.e., turning vehicles waiting for pedestrians to cross side streets
            before completing the turn. Does not directly model delay due to deceleration, although fixed deceleration/reacceleration delay
            may be specified. This may be passed to the constructor or specified in tt_settings (tt_settings value will be used if
            no value is passed to the constructor).
    
        Requires turn demand (veh/hr), left turn dummy (0=right, 1=left), number of turn lanes, ped demand (peds/hr),
        crossing distance (lane-equivalents), exit lanes (lanes),
        cycle length (sec), veh turn green phase (sec), ped crossing phase (sec) [assumed to maximize overlap]

        OPTIONAL: fixed delay experienced by bus due to turning vehicles decelerating (includes time lost to bus decelerating
        and reaccelerating) (seconds)

        .. note:: Generally, an instance of this class will be used to model delay to a through-moving bus or tram
            experienced as a result of vehicles slowing or queuing to turn from the bus or tram's lane onto a side street.
                *   *Example*: If the bus/tram operates in the rightmost lane and left turns are not allowed from that lane, then an instance
                    would be used to model delay due to right-turning vehicles at a given intersection.
                *   *Example*: If the bus/tram operates in a single-lane, one-way street that allows both right and left turns from the same
                    lane, then two instances of this class would be required per intersection, to model delay due to right-turning and
                    left-turning vehicles, respectively.

        .. note:: This class uses formulas to estimate delay that are based on regressions of data collected in San Francisco. Left-turn data
            were only collected for one-way to one-way streets. The same patterns may or may not hold in other locations and cases.

        :param veh_demand: hourly rate of vehicle turns from the bus's through street onto side streets (in the applicable direction)
        :param left_turn: 0 or False to model right-turning vehicles; 1 or True to model left-turning vehicles
        :param turn_lanes: number of lanes from which vehicles making the specified movement may turn (including the lane the bus/tram operates in)
        :param ped_demand: hourly rate of pedestrians crossing the side street onto which vehicles turn
        :param curb_to_curb: curb-to-curb crossing distance for pedestrians crossing the side street, in lane-equivalents.
            (auto or bus lane = +1, parking lane = +1, bike lane = +0.5)
        :param exit_lanes: number of lanes on side street via which turning vehicles can exit the intersection. generally includes
            both mixed-use and restricted lanes that operate in the direction of turning vehicle travel. (e.g., vehicles may exit the intersection
            in a bus lane and then transition to a mixed-use lane shortly after passing the crosswalk.)
        :param cycle_len: length of a full signal cycle, in seconds
        :param turn_phase: duration of signal phase during which turning vehicles may turn (protected and/or permissive)
        :param ped_phase: duration of signal phase during which pedestrians may cross the side street
        :param fixed_delay: fixed delay experienced by bus due to turning vehicles decelerating (includes time lost to bus decelerating
            and reaccelerating) (seconds)

        .. note:: Time values will be rounded to the nearest second.

        .. note:: ``ped_phase`` and ``turn_phase`` are assumed to have maximal overlap; i.e., an intersection is assumed to have either a
            leading pedestrian interval, or a protected green phase, but not both.

        .. note:: Assumes one green phase per signal cycle. Multiple green phases may be approximated by providing the combined green time.

    """
    def __init__(self, veh_demand, left_turn, turn_lanes, ped_demand, curb_to_curb, exit_lanes, cycle_len, turn_phase, ped_phase, fixed_delay=TURN_VEH_DECEL_ACCEL_PENALTY):
        if(VERBOSITY > 3):
            print time.ctime() + ': TurningPoint: creating turning point station object,'
            print time.ctime() + ': TurningPoint: veh_demand: ' + str(veh_demand)
            print time.ctime() + ': TurningPoint: left_turn: ' + str(left_turn)
            print time.ctime() + ': TurningPoint: turn_lanes: ' + str(turn_lanes)
            print time.ctime() + ': TurningPoint: ped_demand: ' + str(ped_demand)
            print time.ctime() + ': TurningPoint: curb_to_curb: ' + str(curb_to_curb)
            print time.ctime() + ': TurningPoint: exit_lanes: ' + str(exit_lanes)
            print time.ctime() + ': TurningPoint: cycle_len: ' + str(cycle_len)
            print time.ctime() + ': TurningPoint: turn_phase: ' + str(turn_phase)
            print time.ctime() + ': TurningPoint: ped_phase: ' + str(ped_phase)
        # process input parameters (all must be nonnegative)
        if cycle_len < 0: # not yet supported
            raise AssertionError('TurningPoint: sorry, support for non-signalized intersections not yet implemented')
        assert_numeric(veh_demand, True)
        assert_numeric(1*left_turn, True) # multiply by 1 to accommodate boolean values
        assert_numeric(turn_lanes, True)
        assert_numeric(ped_demand, True)
        assert_numeric(curb_to_curb, True)
        assert_numeric(exit_lanes, True)
        assert_numeric(cycle_len, True)
        assert_numeric(turn_phase, True)
        assert_numeric(ped_phase, True)
        assert_numeric(fixed_delay, True)
        self.veh_demand = Decimal(veh_demand)
        self.left_turn = int(left_turn)
        self.turn_lanes = int(turn_lanes)
        self.ped_demand = Decimal(ped_demand)
        self.curb_to_curb = Decimal(curb_to_curb)
        self.exit_lanes = int(exit_lanes)
        self.cycle_len = int(cycle_len)
        self.turn_phase = int(turn_phase)
        self.ped_phase = int(ped_phase)
        self.decel_accel_delay = int(fixed_delay)
        # validate and process functional parameters
        if self.left_turn not in (0,1):
            raise AssertionError('Error! Turning point delay: Left turn dummy value should be 0 or 1, instead got ' + str(left_turn))
        if self.exit_lanes > self.curb_to_curb:
            raise AssertionError('Error! Turning point delay: Pedestrian crossing distance is less than the number of exit lanes')
        if self.ped_phase > self.cycle_len or self.turn_phase > self.cycle_len:
            raise AssertionError('Error! Turning point delay: Cycle phase exceeds cycle length')
        ped_demand_per_second = self.ped_demand / 3600 # convert per-hour to per-second
        ped_demand_per_cycle = ped_demand_per_second * self.cycle_len 
        self.ped_rate_avg = ped_demand_per_cycle / self.ped_phase # avg num of pedestrians per second crossing the street while walk signal active
        self.ped_count_dist = PoissonDistribution(ped_demand_per_cycle)
        self.max_peds = self.ped_count_dist.max_count()

        self.probability = []
        prob_no_peds = self.ped_count_dist.prob(0)
        self.probability.append(prob_no_peds) # no delay if zero pedestrians
        cum_prob = self.probability[0]

        # calculate distribution of mean delay for turning vehicles
        # regression model: log of delay (log-seconds) (conditional on some delay existing)
        # R-squared = 0.4279, SE = 0.7068
        mean_log_delay_regress = Decimal(1.50447) \
                               + Decimal(-9.89102) * self.left_turn \
                               + Decimal(-0.71690) * self.curb_to_curb \
                               + Decimal(4.98453) * self.exit_lanes \
                               + Decimal(-0.09112) * self.turn_phase \
                               + Decimal(0.07611) * self.cycle_len \
                               + Decimal(1.85646) * self.turn_lanes \
                               + Decimal(-4.86850) * self.exit_lanes / self.turn_lanes
                               # + Decimal(0.66538) * num_peds / self.ped_phase  # this final term will be added in for each num_peds below
        stdev_mean_log_delay_regress = Decimal(0.7068)
                               
        num_peds = 1 # we already captured 0 ped case above
        while num_peds <= self.max_peds and cum_prob < APPROXIMATE_CERTAINTY:
            prob_this_peds = self.ped_count_dist.prob(num_peds)
            # apply regression model
            mean_log_delay_this_peds = mean_log_delay_regress \
                                       + Decimal(0.66538) * num_peds / self.ped_phase
            if mean_log_delay_this_peds <= 0: # log(x*y) = log(x) + log(y)
                max_log_delay = int(math.ceil(Decimal(math.log(MAX_DEVIATIONS)) + stdev_mean_log_delay_regress))
            else: # log(x+y) = log(x) + log(1+exp(log(y)-log(x)))
                max_log_delay = int(math.ceil( \
                                                math.log(mean_log_delay_this_peds) \
                                                + math.log(1 + math.exp( \
                                                                        math.log(MAX_DEVIATIONS * stdev_mean_log_delay_regress) \
                                                                        - math.log(mean_log_delay_this_peds)))))
            delay_this_peds = LognormalDistribution(mean_log_delay_this_peds, stdev_mean_log_delay_regress, max_log_delay)
            delay_sec = 0
            max_delay = delay_this_peds.max_delay()
            if VERBOSITY > 6:
                print time.ctime() + ': TurningPoint: Max delay for ' + str(num_peds) + ' peds is ' + str(max_delay) + ' sec'
            while delay_sec <= max_delay:
                prob_this_delay = prob_this_peds * delay_this_peds.prob(delay_sec)
                while True:
                    try:
                        self.probability[delay_sec] += prob_this_delay
                        cum_prob += prob_this_delay
                        break
                    except IndexError:
                        self.probability.append(Decimal(0))
                delay_sec += 1
            num_peds += 1
            
        # rescale based on cumulative probability
        if(VERBOSITY > 4):
            print time.ctime() + ': TurningPoint: rescaling probabilities, base cumulative probability (for turning veh delay) is ' + str(cum_prob) + ' (should be near 1)'
        scaled_probability = []
        for prob in self.probability:
            scaled_probability.append(prob/cum_prob)
        self.probability = scaled_probability

        # even conditional on peds being present, there is a 39/279 (~14%) chance that no veh delay is experienced.
        # (the regression above was conditional on some delay existing)
        # we'll take this as a haircut off the average vehicle delay probabilities
        prob_no_delay_cond_peds = Decimal(39)/279
        prob_delay_cond_peds = 1 - prob_no_delay_cond_peds
        prob_peds_present = 1 - prob_no_peds
        prob_no_delay_new = prob_no_peds + prob_no_delay_cond_peds * prob_peds_present
        adjusted_probability = []
        adjusted_probability.append(prob_no_delay_new) # this is the empirical probability of no delay
        prob_delay_new = 1 - prob_no_delay_new
        prob_delay_previous = 1 - self.probability[0]
        adjustment_factor = prob_delay_new / prob_delay_previous
        for prob in self.probability:
            adjusted_probability.append(prob * adjustment_factor)
        self.probability = adjusted_probability

        # rescale based on cumulative probability
        if(VERBOSITY > 4):
            cum_prob = Decimal(0)
            for prob in self.probability:
                cum_prob += prob
            print time.ctime() + ': TurningPoint: after adjustment, cumulative probability is ' + str(cum_prob) + ' (should be near 1)'
        scaled_probability = []
        for prob in self.probability:
            scaled_probability.append(prob/cum_prob)
        self.probability = scaled_probability

        # self.probability now reflects the probability distribution for the average delay for turning vehicles within a signal cycle
        # what is the probability of N turning vehicles during a given light cycle? it relates to the demand level:
        cycle_len_hrs = Decimal(self.cycle_len) / 3600
        veh_demand_per_cycle = self.veh_demand * cycle_len_hrs
        num_veh_this_cycle = PoissonDistribution(veh_demand_per_cycle)

        # what is the distribution of delay for individual vehicles? its proportion to average delay varies with the number of vehicles:
        # standard deviation of delay / average delay = 0.14921 * number of vehicles    (R-squared = 0.4083, SE = 0.4311)

        # how does this relate to the delay experienced by the bus? we assume a bus experiences delay equivalent to that of the turning 
        # vehicle directly ahead of it. so if at least one vehicle is ahead of the bus, the distribution of delay is equivalent to the above.
        # there is some chance that no turning vehicles are ahead of the bus, however, in which case there is no delay.
        # the probability that at least one turning vehicle is ahead of the bus is given by:
        # P(turns ahead) = 1 - (0.5)^N    (where N is the number of turning vehicles this cycle)
        if VERBOSITY > 5:
            print time.ctime() + ': TurningPoint: Calculating delay experienced by bus'
        bus_delay_probability = []
        max_veh = num_veh_this_cycle.max_count() # maximum number of vehicles that could be turing in one signal cycle
        if VERBOSITY > 6:
            print time.ctime() + ': TurningPoint: Calculating delay experienced by bus, max turning vehicles: ' + str(max_veh)
        # calculate no vehicle probability separately
        prob_no_veh = num_veh_this_cycle.prob(0)
        cum_prob = prob_no_veh
        bus_delay_probability.append(prob_no_veh)
        for this_num_veh in range(1, 1 + max_veh):
            if cum_prob > APPROXIMATE_CERTAINTY:
                break
            prob_this_num_veh = num_veh_this_cycle.prob(this_num_veh)
            prob_bus_behind_veh = 1 - Decimal(0.5)**this_num_veh
            # if bus is not behind any turning vehicles, there is no delay
            prob_not_behind_turning_vehs = prob_this_num_veh * (1 - prob_bus_behind_veh)
            bus_delay_probability[0] += prob_not_behind_turning_vehs
            cum_prob += prob_not_behind_turning_vehs
            if VERBOSITY > 6:
                print time.ctime() + ': TurningPoint: Calculating delay experienced by bus, ' + str(this_num_veh) + ' turning veh within signal cycle; conditional probability: ' + str(prob_this_num_veh) + ', prior cum. prob: ' + str(cum_prob)
            max_mean_delay_sec = self.max_delay() # maximum average delay for turning vehicles this cycle
            if VERBOSITY > 7:
                print time.ctime() + ': TurningPoint: Calculating delay experienced by bus, ' + str(this_num_veh) + ' turning veh within signal cycle, max delay: ' + str(max_mean_delay_sec)
            for mean_delay_sec in range(1 + max_mean_delay_sec): # what is the delay to the average veh?
                if cum_prob > APPROXIMATE_CERTAINTY:
                    break
                prob_this_mean_delay = self.prob(mean_delay_sec)
                if VERBOSITY > 7:
                    print time.ctime() + ': TurningPoint: Calculating delay experienced by bus, ' + str(this_num_veh) + ' turning veh, mean delay: ' + str(mean_delay_sec) + '; conditional probability: ' + str(prob_this_mean_delay) + ', prior cum. prob: ' + str(cum_prob)
                expected_std_dev = Decimal(0.14921) * this_num_veh * mean_delay_sec
                expected_max_delay = int(math.ceil(mean_delay_sec + MAX_DEVIATIONS * expected_std_dev))
                # we could do something with these to be a bit more rigorous but let's ignore for simplicity.
                # std_dev_std_dev = Decimal(0.4311) * mean_delay_sec
                # max_delay = int(math.ceil(mean_delay_sec + MAX_DEVIATIONS * (expected_std_dev + MAX_DEVIATIONS * std_dev_std_dev)))
                expected_delay_dist = LognormalDistribution.from_mean_stdev(mean_delay_sec, expected_std_dev, expected_max_delay)
                for bus_delay_sec in range(1 + expected_max_delay): # what is the delay actually experienced by the bus?
                    if cum_prob > APPROXIMATE_CERTAINTY:
                        break
                    prob_this_bus_delay = expected_delay_dist.prob(bus_delay_sec)
                    unconditional_prob = prob_this_num_veh * prob_bus_behind_veh * prob_this_mean_delay * prob_this_bus_delay
                    if VERBOSITY > 8:
                        print time.ctime() + ': TurningPoint: Calculating delay experienced by bus, ' + str(this_num_veh) + ' turning veh, mean delay: ' + str(mean_delay_sec) + ', bus delay (excl. decel/accel): ' + str(bus_delay_sec) + ', absolute probability: ' + str(unconditional_prob) + ', cum. prob: ' + str(cum_prob)
                    while True:
                        try:
                            bus_delay_probability[bus_delay_sec + self.decel_accel_delay] += unconditional_prob
                            cum_prob += unconditional_prob
                            break
                        except IndexError:
                            bus_delay_probability.append(Decimal(0))

        if(VERBOSITY > 8):
            print time.ctime() + ': TurningPoint: ostensible cumulative probability is ' + str(cum_prob)
            
        cum_prob = 0
        for prob in bus_delay_probability:
            cum_prob += prob
            
        if(VERBOSITY > 8):
            print time.ctime() + ': TurningPoint: recalculated cumulative probability is ' + str(cum_prob)                
            
        # rescale based on cumulative probability and assign to self.probability
        if(VERBOSITY > 4):
            print time.ctime() + ': TurningPoint: rescaling probabilities, base cumulative probability (for bus delay) is ' + str(cum_prob) + ' (should be near 1)'
        scaled_probability = []
        i = 0
        for prob in bus_delay_probability:
            scaled_probability.append(prob/cum_prob)
            if(VERBOSITY > 8):
                print time.ctime() + ': TurningPoint: Probability of ' + str(i) + ' sec delay is: ' + str(prob/cum_prob)
                i += 1
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

        


