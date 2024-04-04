import numpy as np

from federatedscope.core.splitters import BaseSplitter


class QuantitySplitter(BaseSplitter):
    """
    This splitter splits dataset following the independent and identically \
    distribution.

    Args:
        client_num: the dataset will be split into ``client_num`` pieces
    """

    def __init__(self, client_num, partition=None):
        BaseSplitter.__init__(self, client_num)
        if partition is None:
            partition = [0.05, 0.10, 0.15, 0.20, 0.5]
        self.partition = partition

        for i in range(1, len(self.partition)):
            self.partition[i] = self.partition[i] + self.partition[i - 1]

        assert len(self.partition) == self.client_num

    def __call__(self, dataset, **kwargs):
        from torch.utils.data import Dataset, Subset

        length = len(dataset)
        index = [x for x in range(length)]
        np.random.shuffle(index)

        # Calculate the number of samples for each partition
        partition_sizes = [int(p * length) for p in self.partition]

        idx_slice = np.split(np.array(index), np.array(partition_sizes[:-1]))

        if isinstance(dataset, Dataset):
            data_list = [Subset(dataset, idxs) for idxs in idx_slice]
        else:
            data_list = [[dataset[idx] for idx in idxs] for idxs in idx_slice]

        return data_list
