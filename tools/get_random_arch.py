import logging
import numpy as np
import os
import os.path as osp
import logging
import time


widen_factor_range = [0.125, 0.25, 0.375, 0.5]
deepen_factor_range = [0.33]
times = time.strftime('%Y%m%d_%H%M%S', time.localtime())
log_dir='./get_random_arch'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logfile = os.path.join(log_dir,'{}.log'.format(times))
logging.basicConfig(filename=logfile, level=logging.INFO)

for k in range(10):
    arch = {}
    arch['widen_factor_backbone'] = []
    arch['deepen_factor'] = []
    arch['widen_factor_neck'] = []
    for i in range(5):
        arch['widen_factor_backbone'].append(
            widen_factor_range[np.random.randint(0, len(widen_factor_range))])  # todo [0,1]
    for i in range(4):
        arch['deepen_factor'].append(deepen_factor_range[np.random.randint(0, len(deepen_factor_range))])
    for i in range(8):
        arch['widen_factor_neck'].append(
            widen_factor_range[np.random.randint(0, len(widen_factor_range))])
    arch['widen_factor_neck_out'] = widen_factor_range[np.random.randint(0, len(widen_factor_range))]
    logging.info(arch)
