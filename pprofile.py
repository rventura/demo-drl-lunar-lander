#! /usr/bin/env python3

import sys
import pstats
from pstats import SortKey

p = pstats.Stats("profile.out")
p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(25)
