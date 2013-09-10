""" Script to calculate cumulative distribution of delay given various elements of delay
    teo@sfcta.org, 8/26/2013
"""

import math, decimal, time
from decimal import Decimal

VERBOSITY = 5                               # higher values will produce more feedback (0-10)
decimal.getcontext().prec = 6               # decimal precision
MAX_DEVIATIONS = 5                          # in a normal or poisson distribution, maximum deviations from the mean that will be analyzed.
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
                * files are assumed to be formatted as saved by Excel

input files:
                TrafficSignals[scenario_name].csv
                    - columns: Name, Cycle time, Green time, Fixed delay
                    - values in seconds
                StopSigns[scenario_name].csv
                    - columns: Name, Fixed delay, Wait probability, Max wait
                    - values in seconds (except probability)
                PedXings[scenario_name].csv
                    - columns: Name, Delay probability, Max delay (seconds)
                StationsStops[scenario_name].csv
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



class HeadwayDistribution:
    """ Duration object for headways
        Requires mean and standard deviation of headways, and maximum possible headway, in minutes
        Note: Only calculates distribution of headways up to APPROXIMATE_CERTAINTY
    """
    def __init__(self, mu, sigma, max_headway):
        if(VERBOSITY > 3):
            print time.ctime()
            print 'creating headway distribution object,'
            print 'average delay: ' + str(mu)
            print 'standard deviation: ' + str(sigma)
        assert_numeric(mu)
        assert_numeric(sigma)
        assert_numeric(max_headway)
        self.mean = Decimal(60*mu)
        self.variance = Decimal(60*sigma)**2
        self.probability = []
        self.max = 60*max_headway

        # normal distribution
        headway_sec = 0
        cum_prob = Decimal(0)
        while(headway_sec <= self.max):
            this_prob = Decimal( math.exp(-1 * ((headway_sec-self.mean) ** 2) / (2*self.variance)) / math.sqrt(2 * Decimal(math.pi) * self.variance) )
            self.probability.append(this_prob)
            cum_prob += this_prob
            if(VERBOSITY > 9):
                print 'assigned headway distribution: ' + str(headway_sec) + ', probability: ' + str(this_prob)
                print 'cumulative probability now: ' + str(cum_prob)
            headway_sec += 1

        # rescale based on cumulative probability
        scaled_probability = []
        for prob in self.probability:
            scaled_probability.append(prob/cum_prob)
        self.probability = scaled_probability


    def prob(self, headway_sec):
        try:
            return self.probability[headway_sec]
        except IndexError:
            return -1

    def max_duration(self):
        return -1 + len(self.probability)


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
            return -1

    def max_delay(self):
        return -1 + len(self.probability)


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
        max_headway = headway_obj.max_duration()
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
            return -1

    def max_delay(self):
        return -1 + len(self.probability)



class CumulativeDistribution:
    """ Calculates the cumulative distribution of delay, given components of delay
        Requires a set of one or more objects which must each have a prob() function
        that returns the probability of delay, in seconds, as passed to the function,
        and a max_delay() function which returns the maximum possible delay, in seconds
    """
    def __init__(self, delay_obj_list):
        if(VERBOSITY > 3):
            print time.ctime() + ': creating cumulative distribution object'
        self.probability = []
        self.partial_probs = []
        for trial in range(len(delay_obj_list)):
            delay_obj = delay_obj_list[trial]
            max_delay = delay_obj.max_delay()
            self.partial_probs.append([])
            delay_sec = 0
            cum_prob = Decimal(0)
            while(delay_sec <= max_delay):
                this_prob = delay_obj.prob(delay_sec)
                if(this_prob < 0):
                    break
                self.partial_probs[trial].append(this_prob)
                cum_prob += this_prob
                delay_sec += 1
        self.num_partials = len(self.partial_probs)
        self.calc_cum_probs()

    def calc_cum_probs(self, existing_probs=[1]):
        new_probs = []
        try:
            incr_probs = self.partial_probs.pop(0)
        except IndexError:
            self.probability = existing_probs
            return
        for delay_sec in range(len(existing_probs)):
            for incr_delay in range(len(incr_probs)):
                this_delay = delay_sec + incr_delay
                this_prob = existing_probs[delay_sec] * incr_probs[incr_delay]
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
            return -1
        
    def max_delay(self):
        return -1 + len(self.probability)



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
    TrafficSignals_path = os.path.join(dir_path, 'TrafficSignals' + SCENARIO + '.csv')
    StopSigns_path      = os.path.join(dir_path, 'StopSigns' + SCENARIO + '.csv')
    PedXings_path       = os.path.join(dir_path, 'PedXings' + SCENARIO + '.csv')
    StationsStops_path  = os.path.join(dir_path, 'StationsStops' + SCENARIO + '.csv')

    # Initial notification
    if(VERBOSITY > 6):
        print 'BEGIN RUN'
        print time.ctime()
        print "APPROXIMATE_CERTAINTY: " + str(APPROXIMATE_CERTAINTY)
        print "APPROXIMATE_ZERO: " + str(APPROXIMATE_ZERO)


    # Process traffic signal info
    incl_TrafficSignals = True
    try:
        with open(TrafficSignals_path, 'rb') as TrafficSignals_csv:
            TrafficSignals = []
            signal_reader = csv.reader(TrafficSignals_csv, dialect='excel')
            header = signal_reader.next() # throw away the header as it doesn't contain data
            for row in signal_reader:
                name        = row[0]
                cycle_time  = Decimal(row[1])
                green_time  = Decimal(row[2])
                fixed_time  = Decimal(row[3])
                TrafficSignals.append(TrafficSignal(cycle_time, green_time, fixed_time))
                if(VERBOSITY > 1):
                    print time.ctime() + ': Read traffic signal: ' + name
        TotalTrafficSignalDelay = CumulativeDistribution(TrafficSignals)
        outfile_path = os.path.join(dir_path, 'TrafficSignals' + SCENARIO + '_cumulative.csv')
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
                name        = row[0]
                fixed_delay = Decimal(row[1])
                num_doors   = Decimal(row[2])
                hrly_board  = Decimal(row[3])
                hrly_alight = Decimal(row[4])
                board_pace  = Decimal(row[5])
                alight_pace = Decimal(row[6])
                headway_avg = Decimal(row[7])
                headway_sd  = Decimal(row[8])
                stop_reqd   = row[9]
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
                headway_obj = HeadwayDistribution(headway_avg, headway_sd, max_calculable_headway)
                TrainStations.append(TrainStation(fixed_delay, num_doors, hrly_board, hrly_alight, board_pace, alight_pace, headway_obj, stop_reqd))
                if(VERBOSITY > 1):
                    print time.ctime() + ': Calculated delay for station / stop: ' + name
        TotalStationDelay = CumulativeDistribution(TrainStations)
        outfile_path = os.path.join(dir_path, 'StationsStops' + SCENARIO + '_cumulative.csv')
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
    if(incl_TrafficSignals):
        delay_component_dists.append(TotalTrafficSignalDelay)
        if(len(delay_component_str) == 0):
            delay_component_str += 'Traffic signals'
        else:
            delay_component_str += '; Traffic signals'
    if(incl_StopSigns):
        delay_component_dists.append(TotalStopSignDelay)
        if(len(delay_component_str) == 0):
            delay_component_str += 'Stop signs'
        else:
            delay_component_str += '; Stop signs'
    if(incl_PedXings):
        delay_component_dists.append(TotalPedXingDelay)
        if(len(delay_component_str) == 0):
            delay_component_str += 'Pedestrian crossings'
        else:
            delay_component_str += '; Pedestrian crossings'
    if(incl_StationsStops):
        delay_component_dists.append(TotalStationDelay)
        if(len(delay_component_str) == 0):
            delay_component_str += 'Boarding and alighting'
        else:
            delay_component_str += '; Boarding and alighting'
        
    CumulativeCorridorDelay = CumulativeDistribution(delay_component_dists)
    outfile_path = os.path.join(dir_path, 'TotalDelay' + SCENARIO + '_cumulative.csv')
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
            print time.ctime() + ': Wrote cumulative boarding & alighting delay: ' + outfile_path
            
    # Finish up
    if(VERBOSITY > 0):
        print time.ctime() + ': DELAY DISTRIBUTION CALCULATION - COMPLETED SUCCESSFULLY!'
