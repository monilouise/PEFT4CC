import pandas as pd
from torch.utils.data import Sampler
import random
import logging

logger = logging.getLogger(__name__)

def calc_obf0(ma, target_th, l0, m):
  if ma > target_th:
    return (l0*((m**ma - m**target_th)/(m - m**target_th))) + 1
  else:
    return 1

def calc_obf1(ma, target_th, l1, m):
  if ma <= target_th:
    return (l1*((m**(target_th - ma) - 1)/(m**target_th - 1))) + 1
  else:
    return 1

def update_orb(preds, target_th, l0, l1, m):
  ma = sum(preds)/len(preds)
  obf0 = calc_obf0(ma, target_th, l0, m)
  obf1 = calc_obf1(ma, target_th, l1, m)
  return obf0, obf1

def calculate_target_th(training_pool):
  df = pd.DataFrame(training_pool)
  return df['is_buggy_commit'].mean()

class SkewedRandomSampler(Sampler):
    def __init__(self, training_set, num_samples=None):
        self.data_source = training_set
        self._num_samples = num_samples
        self.obf0 = 1
        self.obf1 = 1

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        return self.skewed_sample()

    def __len__(self):
        return self.num_samples

    def pick_random(self, label):
        result = None

        while result is None:
            r = random.randint(0, len(self.data_source) - 1)
            if self.data_source[r][3] == label:
                result = r

        return result

    def skewed_sample(self):
        s = []
        logger.info(f'obf0 = {self.obf0}')
        logger.info(f'obf1 = {self.obf1}')

        for _ in range(self.num_samples):
            r = random.randint(0, int(self.obf0 + self.obf1))
            if r < self.obf0:
                s.append(self.pick_random(0))
            else:
                s.append(self.pick_random(1))

        logger.info(f"Skewed sample: {s}")
        return iter(s)