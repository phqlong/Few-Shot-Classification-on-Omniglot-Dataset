"""
Copy from torchmeta.utils.data.sampler.CombinationRandomSampler
Add num_samples parameter to __init__ for select num_samples from dataset
"""

import random
import warnings
from itertools import combinations
from torch.utils.data.sampler import RandomSampler

from torchmeta.utils.data.dataset import CombinationMetaDataset


class CombinationRandomSampler(RandomSampler):
    def __init__(self, data_source, num_samples):
        if not isinstance(data_source, CombinationMetaDataset):
            raise TypeError('Expected `data_source` to be an instance of '
                            '`CombinationMetaDataset`, but found '
                            '{0}'.format(type(data_source)))
        # Temporarily disable the warning if the length of the length of the 
        # dataset exceeds the machine precision. This avoids getting this
        # warning shown with MetaDataLoader, even though MetaDataLoader itself
        # does not use the length of the dataset.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            super(CombinationRandomSampler, self).__init__(data_source,
                                                           replacement=True,
                                                           num_samples=num_samples)

    def __iter__(self):
        num_classes = len(self.data_source.dataset)
        num_classes_per_task = self.data_source.num_classes_per_task
        for _ in combinations(range(num_classes), num_classes_per_task):
            yield tuple(random.sample(range(num_classes), num_classes_per_task))
