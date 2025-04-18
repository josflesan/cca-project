"""
Scheduling Policy for batch jobs in heterogeneous cluster (Part 3)
"""

import subprocess
import yaml
import time
from pathlib import Path

jobs = ["blackscholes.yaml", "canneal.yaml", "dedup.yaml", "ferret.yaml", "freqmine.yaml", "radix.yaml", "vips.yaml"]
