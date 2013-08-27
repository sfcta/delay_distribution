""" Script to calculate cumulative distribution of delay given various elements of delay
    teo@sfcta.org, 8/26/2013
"""

VERBOSITY = 5

USAGE = """

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
                    - columns: Name, Delay probability, Max delay
                StationsStops[scenario_name].csv
                    - columns: Name, Access doors, Hourly boardings,
                               Hourly alightings
                    - values count of doors and rates per hour                
"""

def assert_numeric(obj):
    if not isinstance(obj, (int, long, float)):
        raise AssertionError('Data error: ' + str(obj) + ' is not a number')

class TrafficSignal:
    """ Delay object for traffic signals
        Requires cycle time, green time, and any fixed delay beyond waiting for green (all in seconds)
    """
    def __init__(self, cycle_time, green_time, fixed_delay=0):
        assert_numeric(cycle_time)
        assert_numeric(green_time)
        assert_numeric(fixed_delay)
        if(VERBOSITY > 3):
            print 'creating signal object,'
            print 'cycle time: ' + str(cycle_time)
            print 'green time: ' + str(green_time)
            print 'fixed time: ' + str(fixed_delay)
        self.cycle = cycle_time
        self.green = green_time
        self.addl_fixed = fixed_delay
        self.probability = []
        delay_sec = 0
        cum_prob = 0
        while cum_prob < 1:
            if(VERBOSITY > 8):
                print 'cum_prob=' + str(cum_prob)
                print 'delay_sec=' + str(delay_sec)
            if(delay_sec < self.addl_fixed):
                self.probability.append(0)      # impossible to have delay less than fixed
            elif(delay_sec == self.addl_fixed):
                pgreen = 1.0*self.green/self.cycle # caveat: doesn't consider progression.  Assumes complete independence among signals, which is probably appropriate given that there is at least one stop between signals in each scenario.
                self.probability.append(pgreen) # probability of minimum delay (arrive at green light)
                cum_prob += pgreen
            else:
                self.probability.append(1.0/self.cycle)
                cum_prob += 1.0/self.cycle
            delay_sec += 1

    def prob(self, delay_sec):
        try:
            return self.probability[delay_sec]
        except IndexError:
            return 0

    def max_delay(self):
        return -1 + len(self.probability)


class CumulativeDistribution:
    """ Calculates the cumulative distribution of delay, given components of delay
        Requires a set of one or more objects which must each have a prob() function
        that returns the probability of delay, in seconds, as passed to the function
    """
    def __init__(self, delay_obj_list, max_delay=86400):
        self.partial_probs = []
        self.max_cum_delay = max_delay
        for trial in range(len(delay_obj_list)):
            delay_obj = delay_obj_list[trial]
            self.partial_probs.append([])
            delay_sec = 0
            cum_prob = 0
            while cum_prob < 1:
                if(delay_sec >= self.max_cum_delay):
                    print 'Coding error! Cumulative delay of ' + delay_sec + ' seconds exceeds maximum.'
                    return
                this_prob = delay_obj.prob(delay_sec)
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
            if(delay_sec >= self.max_cum_delay):
                print 'Coding error! Cumulative delay of ' + delay_sec + ' seconds exceeds maximum.'
                return
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
            return 0
        
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

    # Process traffic signal info
    incl_TrafficSignals = True
    try:
        with open(TrafficSignals_path, 'rb') as TrafficSignals_csv:
            TrafficSignals = []
            signal_reader = csv.reader(TrafficSignals_csv, dialect='excel')
            header = signal_reader.next() # throw away the header as it doesn't contain data
            for row in signal_reader:
                name        = row[0]
                cycle_time  = float(row[1])
                green_time  = float(row[2])
                fixed_time  = float(row[3])
                TrafficSignals.append(TrafficSignal(cycle_time, green_time, fixed_time))
                if(VERBOSITY > 1):
                    print 'Read traffic signal: ' + name
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
                print 'Wrote cumulative traffic signal delay: ' + outfile_path
    except IOError:
        print 'No traffic signal file found: ' + TrafficSignals_path
        print 'Excluding traffic signal delay from analysis'
        incl_TrafficSignals = False

    # Process stop sign info
    incl_StopSigns = False

    # Process ped xing info
    incl_PedXings = False

    # Process transit stations / stops info
    incl_StationsStops = False

    # Calculate total distribution

    # Finish up
    if(VERBOSITY > 0):
        print 'DELAY DISTRIBUTION CALCULATION - COMPLETED SUCCESSFULLY!'
