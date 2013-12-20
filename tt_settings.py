""" Settings for travel time distribution tool.
        *   Constants that affect the precision and run time of the script are set here
        *   The VERBOSITY constant, which affects debug output, is also set here

    Authors:
        teo@sfcta.org, 12/19/2013
"""


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
@@@ 1. HOUSEKEEPING
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import decimal
from decimal import Decimal


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
@@@ 2. SETTINGS
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# CONSTANT: INPUT FIXED DELAY DUE TO DECELERATION AND ACCELERATION OF BUS CAUSED BY SLOWING RIGHT-TURNING VEHICLES
# VALUE IS A FIXED PENALTY IN SECONDS ANY TIME AT LEAST ONE VEHICLE EXECUTES A RIGHT TURN IN FRONT OF THE BUS
# MUST BE AN INTEGER VALUE
TURN_VEH_DECEL_ACCEL_PENALTY = int(0)

# the following inputs must be integer values. these can be modified to meet your needs.
VERBOSITY = 7                               # higher values will produce more feedback (0-10)
decimal.getcontext().prec = 6               # decimal precision (number of sig figs)
MAX_DEVIATIONS = 5                          # in a normal or lognormal distribution, maximum deviations from the mean that will be analyzed.
                                            # in a poisson distribution, maximum variances from the mean (multiple of the mean) that will be analyzed.
                                            # note: in normal: 3 deviations include 0.997 of values; 4 include 0.99994; 5 include 0.9999994; 6 include 0.999999998

# do not modify
APPROXIMATE_ZERO = Decimal(0.1)**(decimal.getcontext().prec)
APPROXIMATE_CERTAINTY = 1-APPROXIMATE_ZERO
                                            

