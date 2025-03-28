import glob, os
import sys

import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy import constants as const
import astropy.units as u

from .utils import register_custom_filters_on_speclite

from .simulator import SevenDT
from .const import *
from .filter import Filter