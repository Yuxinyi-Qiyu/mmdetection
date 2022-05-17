# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import warnings

import matplotlib.pyplot as plt
import scipy.stats as stats

ap_gt = [
0.707,
0.680,
0.669,
0.680,
0.711,
    0.629,
    0.641,
    0.687,
    0.673,
    0.700,

]
ap_supernet = [
    0.767,
    0.781,
    0.776,
    0.771,
    0.779,
    0.724,
    0.732,
    0.786,
    0.745,
    0.801,
]
tau, p_value = stats.kendalltau(ap_supernet, ap_gt)
print(tau, p_value)





