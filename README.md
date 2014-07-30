delay_distribution
==================

Creates distributions of travel time delay

Full Documentation is available at http://sfcta.github.io/delay_distribution/

# Quick Start

## Setup inputs files

**Input file info:**

- one or more files may be provided for each scenario
- all files should contain a header row, with data beginning on the second row
- files are assumed to be formatted in CSV as saved by Excel
- CSVs are processed based on column position, not column name.  *do not re-arrange columns!*

**Input files:**

`BaseTravelTime_[scenario_name].csv`
 
- Allows specification of base travel time (independent of any modeled delay), as fixed or normally-distributed(normally-distributed travel times will not include any negative values, however, and will therefore reflect rescaledpseudo-normal distributions). The name column is for convenience only and will be ignored.
- required columns:
 - `Name`  Segment name
 - `Average travel time`
 - `Standard deviation`
- all values in seconds

`TravelTimeAdjustments_[scenario_name].csv`

- Allows specification of adjustments to travel time (independent of any modeled delay), as the probabilities of any given adjustments, in seconds. For example, the first row of data may specify the probability of a 0-second adjustment;the second row may specify the probability of a 1-second adjustment; and so on. Negative values are allowed. Duplicate
entries will be double-counted. Probabilities should add to 1 (but will be rescaled otherwise).
- required columns:
 - `delay_sec`
 - `probability`
- can specify additional files as:
                    
 `TravelTimeAdjustments_[scenario_name]_2.csv`
 `TravelTimeAdjustments_[scenario_name]_3.csv`

                        
`BinomialDistributions_[scenario_name].csv`

- Allows specification of binomially-distributed delays (i.e., where there is a fixed probability of a fixed delay
    across multiple analysis zones). Multiple zone groupings or delay types may be specified (in separate rows).
    The name column is for convenience only and will be ignored.
- required columns:
 - `Name` Segment name
 - `n (number of trials)`
 - `p (probability of encounter at each trial)`
 - `delay per encounter (seconds)`
 
`MultinomialDistributions_[scenario_name].csv`

- Allows specification of multinomially-distributed delays (i.e., where there is a fixed probability of stochastically-distributed delay
    across multiple analysis zones). Multiple zone groupings or delay types may be specified (in separate rows).
    The name column is for convenience only and will be ignored.
- required columns:
 - `Name` Segment name
 - `n (number of trials)`
 - `p (probability of encounter at each trial)`
 - `mean of delay per encounter (sec)`
 - `standard deviation of delay per encounter (sec)`
 - `distribution of delay per encounter ("normal" or "lognormal")`
 
`TurningVehiclesAnalytics_[scenario_name].csv`

- Allows specification of parameters related to delay caused by turning vehicles. This specification
    uses regression-based estimates to predict delay based on the data provided.
- required columns:
 - `Turning Point Set`
 - `Number of Turning Points`
 - `turns_per_hr` (turn demand per hour at each point in set),
 - `turn_dir` ("R" or "Right" for right turn; "L"/"Left" if left),
 - `num_turn_lanes` (number of turn lanes at each point),
 - `peds_per_hr` (rate of pedestrians walking parallel to traffic
  that might delay turning vehicles, count per hour),
 - `crossing_dist` (number of lane widths peds cross curb to curb 
  incl. parking and contraflow lanes; bike lanes ~ 0.5 lane)
 - `exit_lanes` (number of travel lanes into which turning vehicles
   could turn, incl. bus lanes but not contraflow or parking),
 - `cycle_len` (length of signal cycle in sec; -1 for no signal),
 - `turn_phase` (total length of green phase for turning vehs, sec),
 - `ped_phase` (total length of walk phase for peds, sec)
- Turning Point Set is a string (e.g.: 'A') that refers to
  a set of turning points with identical delay distributions
- Number of Turning Points is the number of turning points
  in the set
- turn_phase and ped_phase are assumed to have maximal overlap

`TurningVehiclesDetails_[scenario_name].csv`

- Allows specification of delay caused by turning vehicles. This specification takes the fixed
    delay per point (conditional on delay) to predict overall delay.
- required columns:
 - `Turning Point Set`
 - `Number of Turning Points`
 - `delay_sec (per point)`
 - `probability (per point)`
- Turning Point Set is a string (e.g.: 'A') that refers to
  a set of turning points with identical delay distributions
  NOTE that all data for a given set must be grouped
  consecutively (cannot give data for A, then B, then A)
- Number of Turning Points is the number of turning points in the set
- delay_sec is a possible delay, in seconds, for turning
  points in the present set
- probability is the probibility of delay_sec delay for
  turning points in the present set
                      
`TrafficSignals_[scenario_name].csv`

- Allows specification of traffic signal attributes to predict delay due to red lights along
    the study corridor. The name column is for convenience only and will be ignored.
- required columns:
 - `Name` Segment name
 - `Cycle time`
 - `Green time`
 - `Fixed delay`
- values in seconds

`StationsStops_[scenario_name].csv`

- Allows specification of train station or bus stop information to predict delay due to
    dwell time, including boarding and alighting time. The name column is for convenience only and will be ignored.
- required columns:
 - `Name` Segment name
 - `Fixed delay (seconds)`
 - `Number of doors`
 - `Hourly boardings`
 - `Hourly alightings`
 - `Boarding time per passenger per door (seconds)`
 - `Alighting time per passenger per door (seconds)`
 - `Mean headway (minutes)`
 - `Standard deviation of headway (minutes)`
 - `Stop requirement`

               1 or TRUE or YES if stop is required

               0 or FALSE or NO if stop can be skipped when no pax wish to board or alight

## Setup Batch File

Inside `CalculateDelayDistributions.bat`:


    :: Directory where input files live
    set RUNDIR=.

    :: Directory where "generate_delay_distribution.py" sits.  Usually one up from this batch file.
    set SCRIPTDIR=..

    :: List of scenario names that input files are labelled with.
    set ALLSCENARIOS=Scenario1 Scenario2 Scenario3



Docs are managed using Sphinx and posted via the gh-pages branch. For details on this see:
http://daler.github.io/sphinxdoc-test/includeme.html