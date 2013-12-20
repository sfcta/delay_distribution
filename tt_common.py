""" Common functions and resources for travel time distribution tool.

    Authors:
        teo@sfcta.org, 12/19/2013
"""


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
@@@ 1. MODULE IMPORTS
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import math, decimal, time
from decimal import Decimal
from tt_settings import *

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
@@@ 2. GLOBAL FUNCTIONS
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def assert_numeric(obj, non_neg=False):
    if not isinstance(obj, (int, long, float, Decimal)):
        raise AssertionError('Data error: ' + str(obj) + ' is not a number')
    if(non_neg): # number must be non-negative
        if(obj < 0):
            raise AssertionError('Data error: ' + str(obj) + ' is less than zero')

def assert_decimal(obj, non_neg=False):
    if not isinstance(obj, Decimal):
        if not isinstance(obj, (int, long, float)):
            raise AssertionError('Data error: ' + str(obj) + ' is not a number')
        raise AssertionError('Coding error: ' + str(obj) + ' has not been converted to Decimal')
    if(non_neg): # number must be non-negative
        if(obj < 0):
            raise AssertionError('Data error: ' + str(obj) + ' is less than zero')


