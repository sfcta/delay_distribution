"""
    Delay Distribution Generator (generate_delay_distribution)
    ==========================================================
    
    Calculates cumulative distribution of delay
    or travel time given various elements of delay or travel time components. 
    Accurate to 1-second resolution. Output is in the form of a CSV file which
    provides the probabilities of duration for each possible value of time, in
    seconds. Implements 'Trial Bus' statistical technique.

    Data must be provided in the form of CSV files (see details below).

    This python file is organized as follows:
        1.  Housekeeping - importing packages, USAGE parameter.
        2.  Travel time distributions - importing classes for all components of travel time or delay
        3.  Process data - read in CSV file input data; process data and provide outputs

    This script accepts the input as provided in CSV files as follows.
    ------------------------------------------------------------------
        USAGE: ``python generate_delay_distribution.py [scenario_name] spec_dir``

        :scenario_name: *(OPTIONAL)* name of scenario, specifies input and output nomenclature
        :spec_dir: path to directory containing csv files with delay information

        **Input file info:**
                * one or more files may be provided for each scenario
                * all files should contain a header row, with data beginning
                  on the second row
                * files are assumed to be formatted in CSV as saved by Excel

        **Input files:**
                ``BaseTravelTime_[scenario_name].csv``
                    - Allows specification of base travel time (independent of any modeled delay), as fixed or normally-distributed
                        (normally-distributed travel times will not include any negative values, however, and will therefore reflect rescaled
                        pseudo-normal distributions). The name column is for convenience only and will be ignored.
                    - columns
                        - Name,
                        - Average travel time,
                        - Standard deviation
                    - values in seconds
                ``TravelTimeAdjustments_[scenario_name].csv``
                    - Allows specification of adjustments to travel time (independent of any modeled delay), as the probabilities of any
                        given adjustments, in seconds. For example, the first row of data may specify the probability of a 0-second adjustment;
                        the second row may specify the probability of a 1-second adjustment; and so on. Negative values are allowed. Duplicate
                        entries will be double-counted. Probabilities should add to 1 (but will be rescaled otherwise).
                    - columns
                        - delay_sec,
                        - probability
                    - can specify additional files as::
                    
                        TravelTimeAdjustments_[scenario_name]_2.csv
                        TravelTimeAdjustments_[scenario_name]_3.csv
                        etc.
                        
                ``BinomialDistributions_[scenario_name].csv``
                    - Allows specification of binomially-distributed delays (i.e., where there is a fixed probability of a fixed delay
                        across multiple analysis zones). Multiple zone groupings or delay types may be specified (in separate rows).
                        The name column is for convenience only and will be ignored.
                    - columns
                        - Name,
                        - n (number of trials),
                        - p (probability of encounter at each trial),
                        - delay per encounter (seconds)
                ``MultinomialDistributions_[scenario_name].csv``
                    - Allows specification of multinomially-distributed delays (i.e., where there is a fixed probability of stochastically-distributed delay
                        across multiple analysis zones). Multiple zone groupings or delay types may be specified (in separate rows).
                        The name column is for convenience only and will be ignored.
                    - columns
                        - Name,
                        - n (number of trials),
                        - p (probability of encounter at each trial),
                        - mean of delay per encounter (sec),
                        - standard deviation of delay per encounter (sec),
                        - distribution of delay per encounter ("normal" or "lognormal")
                ``TurningVehiclesAnalytics_[scenario_name].csv``
                    - Allows specification of parameters related to delay caused by turning vehicles. This specification
                        uses regression-based estimates to predict delay based on the data provided.
                    - columns
                        - Turning Point Set,
                        - Number of Turning Points,
                        - turns_per_hr (turn demand per hour at each point in set),
                        - turn_dir ("R" or "Right" for right turn; "L"/"Left" if left),
                        - num_turn_lanes (number of turn lanes at each point),
                        - peds_per_hr (rate of pedestrians walking parallel to traffic
                          that might delay turning vehicles, count per hour),
                        - crossing_dist (number of lane widths peds cross curb to curb 
                          incl. parking and contraflow lanes; bike lanes ~ 0.5 lane)
                        - exit_lanes (number of travel lanes into which turning vehicles
                          could turn, incl. bus lanes but not contraflow or parking),
                        - cycle_len (length of signal cycle in sec; -1 for no signal),
                        - turn_phase (total length of green phase for turning vehs, sec),
                        - ped_phase (total length of walk phase for peds, sec)
                    - Turning Point Set is a string (e.g.: 'A') that refers to
                      a set of turning points with identical delay distributions
                    - Number of Turning Points is the number of turning points
                      in the set
                    - turn_phase and ped_phase are assumed to have maximal overlap
                ``TurningVehiclesDetails_[scenario_name].csv``
                    - Allows specification of delay caused by turning vehicles. This specification takes the fixed
                        delay per point (conditional on delay) to predict overall delay.
                    - columns
                        - Turning Point Set,
                        - Number of Turning Points,
                        - delay_sec (per point),
                        - probability (per point)
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
                ``TrafficSignals_[scenario_name].csv``
                    - Allows specification of traffic signal attributes to predict delay due to red lights along
                        the study corridor. The name column is for convenience only and will be ignored.
                    - columns
                        - Name,
                        - Cycle time,
                        - Green time,
                        - Fixed delay
                    - values in seconds
                ``StationsStops_[scenario_name].csv``
                    - Allows specification of train station or bus stop information to predict delay due to
                        dwell time, including boarding and alighting time. The name column is for convenience only and will be ignored.
                    - columns
                        - Name,
                        - Fixed delay (seconds),
                        - Number of doors,
                        - Hourly boardings,
                        - Hourly alightings,
                        - Boarding time per passenger per door (seconds),
                        - Alighting time per passenger per door (seconds),
                        - Mean headway (minutes),
                        - Standard deviation of headway (minutes),
                        - Stop requirement
                            - 1 or TRUE or YES if stop is required
                            - 0 or FALSE or NO if stop can be skipped when no pax wish to board or alight

        .. note:: The following types of delay are stubbed out in the generator file but not yet implemented:
        
            * ``StopSigns_[scenario_name].csv`` [*module not yet available*]
                - columns: Name, Fixed delay, Wait probability, Max wait
                - values in seconds (except probability)
            * ``PedXings_[scenario_name].csv`` [*module not yet available*]
                - columns: Name, Delay probability, Max delay (seconds)
        
    Output CSV files are produced as follows.
    -----------------------------------------
    
        For each type of input CSV provided, for each scenario, an output file will be created with a similar name, but with the ending
        ``_cumulative.csv``. These output files represent the probability distributions of cumulative delay along
        the study corridor due to the delays specified by the relevant input CSV file.close

        For each scenerio, an output file will be created with the name ``TotalDelay_[scenario_name]_cumulative.csv``. These
        output files represent the probability distributions of cumulative delay or cumulative travel time along the study corridor
        due to all specified delays and including base travel time and travel time adjustments.

        All output files have two columns, **Delay** and **Cumulative Delay Probability**. In each row, Delay is a duration of cumulative delay
        or overall travel time, in seconds. Cumulative Delay Probability is the probability of the trial bus experiencing that delay or
        travel time.accept2dyear

        .. note:: Cumulative delay and cumulative travel time are not to be confused with cumulative distribution functions. Although the word
            "cumulative" is used here, to imply the modeling of aggregate delay across analysis zones and delay types, the output distributions are
            probability mass functions (PMFs) that approximate probability density functions (PDFs) on the 1-second scale. Output distributions are NOT
            cumulative distributions (CDFs).


    Authors:
        teo@sfcta.org, 12/20/2013
"""


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
@@@ 1. HOUSEKEEPING
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

from tt_common import *

USAGE = """

Delay Distribution Generator: Calculates cumulative distribution of delay
or travel time given various elements of delay or travel time components. 
Accurate to 1-second resolution. Output is in the form of a CSV file which
provides the probabilities of duration for each possible value of time, in
seconds.

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
                    - note: can specify additional files as:
                      TravelTimeAdjustments_[scenario_name]_2.csv
                      TravelTimeAdjustments_[scenario_name]_3.csv
                      etc.
                BinomialDistributions_[scenario_name].csv
                    - columns: Name, n (number of trials), p (probability of
                      encounter at each trial), delay per encounter (seconds)
                MultinomialDistributions_[scenario_name].csv
                    - columns: Name, n (number of trials), p (probability of
                      encounter at each trial), mean of delay per encounter (sec),
                      standard deviation of delay per encounter (sec),
                      distribution of delay per encounter ("normal" or "lognormal")
                TurningVehiclesAnalytics_[scenario_name].csv
                    - columns: Turning Point Set, Number of Turning Points,
                      turns_per_hr (turn demand per hour at each point in set),
                      turn_dir ("R" or "Right" for right turn; "L"/"Left" if left),
                      num_turn_lanes (number of turn lanes at each point),
                      peds_per_hr (rate of pedestrians walking parallel to traffic
                          that might delay turning vehicles, count per hour),
                      crossing_dist (number of lane widths peds cross curb to curb 
                          incl. parking and contraflow lanes; bike lanes ~ 0.5 lane)
                      exit_lanes (number of travel lanes into which turning vehicles
                          could turn, incl. bus lanes but not contraflow or parking),
                      cycle_len (length of signal cycle in sec; -1 for no signal),
                      turn_phase (total length of green phase for turning vehs, sec),
                      ped_phase (total length of walk phase for peds, sec)
                    - Turning Point Set is a string (e.g.: 'A') that refers to
                      a set of turning points with identical delay distributions
                    - Number of Turning Points is the number of turning points
                      in the set
                    - turn_phase and ped_phase are assumed to have maximal overlap
                TurningVehiclesDetails_[scenario_name].csv
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


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
@@@ 2. TRAVEL TIME DISTRIBUTIONS
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Cumulative distribution to add up travel time across multiple distributions
from cumulative_distribution import CumulativeDistribution

# Generic (non-source-specific) distributions for travel time and event counts
from time_count_distributions_generic \
    import  ArbitraryDistribution, \
            NormalDistribution, \
            LognormalDistribution, \
            BinomialDistribution, \
            MultinomialDistribution, \
            PoissonDistribution

# Source-specific delay distributions
from dd_traffic_signal import TrafficSignal
from dd_turning_point import TurningPoint
from dd_train_station import TrainStation



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
@@@ 3. PROCESS DATA PROVIDED IN CSV FILES
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

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
    TurningVehsA_path   = os.path.join(dir_path, 'TurningVehiclesAnalytics_' + SCENARIO + '.csv')
    TurningVehsD_path   = os.path.join(dir_path, 'TurningVehiclesDetails_' + SCENARIO + '.csv')
    TrafficSignals_path = os.path.join(dir_path, 'TrafficSignals_' + SCENARIO + '.csv')
    StopSigns_path      = os.path.join(dir_path, 'StopSigns_' + SCENARIO + '.csv')
    PedXings_path       = os.path.join(dir_path, 'PedXings_' + SCENARIO + '.csv')
    StationsStops_path  = os.path.join(dir_path, 'StationsStops_' + SCENARIO + '.csv')

    # Initial notification
    if(VERBOSITY > 6):
        print time.ctime() + ': MAIN: BEGIN RUN; APPROXIMATE_CERTAINTY: ' + str(APPROXIMATE_CERTAINTY) + '; APPROXIMATE_ZERO: ' + str(APPROXIMATE_ZERO)

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
                max_calculable_tt = int(math.ceil(avg_time + MAX_DEVIATIONS * stdev_time))
                BaseSegments.append(NormalDistribution(avg_time, stdev_time, max_calculable_tt))
                if(VERBOSITY > 1):
                    print time.ctime() + ': MAIN:  Read base travel time: ' + name
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
                print time.ctime() + ': MAIN: Wrote cumulative base travel time: ' + outfile_path            
    except IOError:
        print time.ctime() + ': MAIN: No base travel time file found: ' + BaseTT_path
        print time.ctime() + ': MAIN: Excluding base travel time and base variation from analysis'
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
                            ProbabilityListIncreasedDelay[delay_sec] += probability
                        except IndexError:
                            while len(ProbabilityListIncreasedDelay <= delay_sec):
                                ProbabilityListIncreasedDelay.append(Decimal(0))
                            ProbabilityListIncreasedDelay[delay_sec] += probability
                    else:
                        try:
                            ProbabilityListReducedDelay.append(Decimal(0))
                            ProbabilityListReducedDelay[-1*delay_sec] += probability
                        except IndexError:
                            while(len(ProbabilityListReducedDelay) <= -1*delay_sec):
                                ProbabilityListReducedDelay.append(Decimal(0))
                            ProbabilityListReducedDelay[-1*delay_sec] += probability                    
                    if(VERBOSITY > 7):
                        print time.ctime() + ': MAIN: Will adjust travel time by ' + str(delay_sec) + ' with probability ' + str(probability)
            TTAdjustments.append(ArbitraryDistribution(ProbabilityListIncreasedDelay,ProbabilityListReducedDelay))
            if(VERBOSITY > 0):
                print time.ctime() + ': MAIN: Calculated travel time adjustments from: ' + thisfile_path
            file_idx += 1
    except IOError:
        if(file_idx == 1):
            print time.ctime() + ': MAIN: No travel time adustment file found: ' + thisfile_path
            print time.ctime() + ': MAIN: Excluding any adjustments from analysis'
            incl_TTAdjust = False
        else:
            print time.ctime() + ': MAIN: File not found: ' + thisfile_path
            print time.ctime() + ': MAIN: Only including travel time adjustments for first ' + str(file_idx - 1) + ' TravelTimeAdjustment files'
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
                    print time.ctime() + ': MAIN: Read binomial distribution: ' + name
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
                print time.ctime() + ': MAIN: Wrote cumulative binomially-distributed delay: ' + outfile_path
    except IOError:
        print time.ctime() + ': MAIN: No binomial delay file found: ' + BinomialDist_path
        print time.ctime() + ': MAIN: Excluding any binomially-distrubted delays from analysis'
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
                max_delay_per_encounter = int(math.ceil(mean + MAX_DEVIATIONS * stdev))
                if(dist_type == 'NORMAL'):
                    delay_per_encounter = NormalDistribution(mean, stdev, max_delay_per_encounter, prob_encounter)
                elif(dist_type == 'LOGNORMAL'):
                    delay_per_encounter = LognormalDistribution.from_mean_stdev(mean, stdev, max_delay_per_encounter, prob_encounter)
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
                print time.ctime() + ': MAIN: Wrote cumulative multinomially-distributed delay: ' + outfile_path
    except IOError:
        print time.ctime() + ': MAIN: No multinomial delay file found: ' + MNDist_path
        print time.ctime() + ': MAIN: Excluding any multinomially-distrubted delays from analysis'
        incl_MNDists = False

    # Process turning vehicles (analytic spec) info
    incl_TurningVehsA = True
    try:
        with open(TurningVehsA_path, 'rb') as TurningVehsA_csv:
            if(VERBOSITY > 1):
                print time.ctime() + ': MAIN: reading in analytical turning vehicle info'
            DelayProbabilityList = []
            TurningPoints = []
            tva_reader = csv.reader(TurningVehsA_csv, dialect='excel')
            header = tva_reader.next() # throw away the header as it doesn't contain data
            for row in tva_reader:
                name            = str(row[0])
                num_points      = int(row[1])
                turns_per_hr    = Decimal(row[2])
                turn_dir        = str(row[3]).upper()
                num_turn_lanes  = int(row[4])
                peds_per_hr     = Decimal(row[5])
                crossing_dist   = Decimal(row[6])
                exit_lanes      = int(row[7])
                cycle_len       = int(row[8])
                turn_phase      = int(row[9])
                ped_phase       = int(row[10])
                if turn_dir in ('L', 'LEFT'): is_left_turn =  1
                elif turn_dir in ('R', 'RIGHT'): is_left_turn = 0
                else: raise AssertionError('Error! Turning vehicles (analytical spec): turn direction "' + turn_dir + '" not valid (expected "L", "R", "LEFT" or "RIGHT")')
                this_turning_point = TurningPoint(turns_per_hr, is_left_turn, num_turn_lanes, peds_per_hr, crossing_dist, exit_lanes, cycle_len, turn_phase, ped_phase)
                for i in range(num_points):
                    TurningPoints.append(this_turning_point)                 
        TotalTurningVehiclesAnalyticsDelay = CumulativeDistribution(TurningPoints)
        if(VERBOSITY > 0):
            print time.ctime() + ': MAIN: Calculated turning vehicle delays from: ' + TurningVehsA_path            
        outfile_path = os.path.join(dir_path, 'TurningVehiclesAnalytics_' + SCENARIO + '_cumulative.csv')
        with open(outfile_path, 'wb') as outfile:
            tva_writer = csv.writer(outfile, dialect='excel')
            delay_sec = 0
            max_delay_sec = TotalTurningVehiclesAnalyticsDelay.max_delay()
            tva_writer.writerow(['CUMULATIVE DELAY DUE TO TURNING VEHICLES, ANALYTICAL SPECIFICATION (' + SCENARIO + ')'])
            tva_writer.writerow(['Delay, sec', 'Cumulative Turning Vehicles (Analytical Spec) Delay Probability, ' + SCENARIO])
            while delay_sec <= max_delay_sec:
                tva_writer.writerow([delay_sec, TotalTurningVehiclesAnalyticsDelay.prob(delay_sec)])
                delay_sec += 1
            if(VERBOSITY > 0):
                print time.ctime() + ': MAIN: Wrote cumulative turning vehicles (analytical spec) delay: ' + outfile_path
    except IOError:
        print time.ctime() + ': MAIN: No turning vehicles analytical specification file found: ' + TurningVehsA_path
        print time.ctime() + ': MAIN: Excluding turning vehicle (analytical spec) delay from analysis'
        incl_TurningVehsA = False

    # Process turning vehicles (detailed spec) info
    incl_TurningVehsD = True
    try:
        with open(TurningVehsD_path, 'rb') as TurningVehsD_csv:
            if(VERBOSITY > 1):
                print time.ctime() + ': MAIN: reading in detailed turning vehicle info'
            DelayProbabilityList = []
            TurningPointSets = []
            latest_set = ''
            latest_num_pts = -1
            tv_reader = csv.reader(TurningVehsD_csv, dialect='excel')
            header = tv_reader.next() # throw away the header as it doesn't contain data
            for row in tv_reader:
                name        = str(row[0])
                num_points  = int(row[1])
                delay_sec   = int(row[2])
                probability = Decimal(row[3])
                if(name == latest_set or latest_num_pts == -1):
                    if(latest_num_pts <> num_points and name == latest_set):
                        raise AssertionError('Error! Mismatched number of turning points (' + str(num_points) + ') for turning set ' + str(name) + ' (expected: ' + str(latest_num_pts) + ')')
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
                        print time.ctime() + ': MAIN: Read turning point set: ' + latest_set
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
                print time.ctime() + ': MAIN: Read turning point set: ' + latest_set                      
        TotalTurningVehiclesDetailsDelay = CumulativeDistribution(TurningPointSets)
        if(VERBOSITY > 0):
            print time.ctime() + ': MAIN: Calculated turning vehicle delays from: ' + TurningVehsD_path            
        outfile_path = os.path.join(dir_path, 'TurningVehiclesDetails_' + SCENARIO + '_cumulative.csv')
        with open(outfile_path, 'wb') as outfile:
            tv_writer = csv.writer(outfile, dialect='excel')
            delay_sec = 0
            max_delay_sec = TotalTurningVehiclesDetailsDelay.max_delay()
            tv_writer.writerow(['CUMULATIVE DELAY DUE TO TURNING VEHICLES, DETAILED SPECIFICATION (' + SCENARIO + ')'])
            tv_writer.writerow(['Delay, sec', 'Cumulative Turning Vehicles (Detailed Spec) Delay Probability, ' + SCENARIO])
            while delay_sec <= max_delay_sec:
                tv_writer.writerow([delay_sec, TotalTurningVehiclesDetailsDelay.prob(delay_sec)])
                delay_sec += 1
            if(VERBOSITY > 0):
                print time.ctime() + ': MAIN: Wrote cumulative turning vehicles (detailed spec) delay: ' + outfile_path
    except IOError:
        print time.ctime() + ': MAIN: No turning vehicles detailed specification file found: ' + TurningVehsD_path
        print time.ctime() + ': MAIN: Excluding turning vehicle (detailed spec) delay from analysis'
        incl_TurningVehsD = False


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
                    print time.ctime() + ': MAIN: Read traffic signal: ' + name
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
                print time.ctime() + ': MAIN: Wrote cumulative traffic signal delay: ' + outfile_path
    except IOError:
        print time.ctime() + ': MAIN: No traffic signal file found: ' + TrafficSignals_path
        print time.ctime() + ': MAIN: Excluding traffic signal delay from analysis'
        incl_TrafficSignals = False


    # Process stop sign info
    try:
        with open(StopSigns_path, 'rb') as StopSigns_csv:
            print time.ctime() + ': MAIN: ERROR: STOP SIGNS NOT YET SUPPORTED.'
            raise AssertionError('Stop Signs not yet supported. Exiting. Please remove StopSigns file before proceeding')
    except IOError:
        incl_StopSigns = False

    # Process ped xing info
    try:
        with open(PedXings_path, 'rb') as PedXings_csv:
            print time.ctime() + ': MAIN: ERROR: UNSIGNALIZED PEDESTRIAN CROSSINGS NOT YET SUPPORTED.'
            raise AssertionError('Ped Xings not yet supported. Exiting. Please remove PedXings file before proceeding')
    except IOError:
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
                    print time.ctime() + ': MAIN: Reading in delay for station / stop: ' + name
                max_calculable_headway = int(math.ceil(headway_avg + MAX_DEVIATIONS * headway_sd))
                headway_obj = NormalDistribution(60*headway_avg, 60*headway_sd, 60*max_calculable_headway)
                TrainStations.append(TrainStation(fixed_delay, num_doors, hrly_board, hrly_alight, board_pace, alight_pace, headway_obj, stop_reqd))
                if(VERBOSITY > 1):
                    print time.ctime() + ': MAIN: Calculated delay for station / stop: ' + name
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
                print time.ctime() + ': MAIN: Wrote cumulative boarding & alighting delay: ' + outfile_path            
    except IOError:
        print time.ctime() + ': MAIN: No transit stations / stops file found: ' + StationsStops_path
        print time.ctime() + ': MAIN: Excluding boarding & alighting delay from analysis'
        incl_StationsStops = False

    # Calculate total distribution of delay
    if(VERBOSITY > 2):
        print time.ctime() + ': MAIN: Calculating total cumulative delay distribution.'
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
    if(incl_TurningVehsA):
        delay_component_dists.append(TotalTurningVehiclesAnalyticsDelay)
        if(len(delay_component_str) > 0):
            delay_component_str += '; '
        delay_component_str += 'Turning vehicles (analytic spec)'
    if(incl_TurningVehsD):
        delay_component_dists.append(TotalTurningVehiclesDetailsDelay)
        if(len(delay_component_str) > 0):
            delay_component_str += '; '
        delay_component_str += 'Turning vehicles (detailed spec)'
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
            print time.ctime() + ': MAIN: Wrote cumulative corridor delay: ' + outfile_path
            
    # Finish up
    if(VERBOSITY > 0):
        print time.ctime() + ': MAIN: DELAY DISTRIBUTION CALCULATION - COMPLETED SUCCESSFULLY!'
