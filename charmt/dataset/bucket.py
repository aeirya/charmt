from torch.utils.data import IterableDataset, DataLoader
from itertools import islice
import random

class BucketStream(IterableDataset):
    def __init__(self, ds, batch_size=32, buffer_size:int=None):
        self.ds = ds
        self.batch_size = batch_size
        self.buffer_size = buffer_size or 50 * batch_size

    def __iter__(self):
        it = iter(self.ds)

        while buf := list(islice(it, self.buffer_size)):
            random.shuffle(buf)
            buf.sort(key=lambda x: max(len(x["input_ids"]), len(x["labels"])))

            for i in range(0, len(buf), self.batch_size):
                yield buf[i:i + self.batch_size]