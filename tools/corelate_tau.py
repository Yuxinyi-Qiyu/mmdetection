# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import warnings

import matplotlib.pyplot as plt
import scipy.stats as stats

ap_gt = [
0.630,
0.755,
0.799,

]
ap_supernet = [
0.618,
0.675,
0.722,
]
tau, p_value = stats.kendalltau(ap_supernet, ap_gt)
print(tau, p_value)





