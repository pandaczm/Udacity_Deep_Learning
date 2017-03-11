__author__ = 'pandaczm'
import collections
import random
import numpy as np

class BatcherConfig:
    def __init__(self, batch_size, num_skips, skip_window):
        self.skip_window = skip_window
        self.num_skips = num_skips
        self.batch_size = batch_size

class Batcher:
    def __init__(self, process_data, batch_config):
        self.data = process_data
        self.batch_config = batch_config
        self.data_index = 0
        self.num_skips = batch_config.num_skips
        self.skip_window = batch_config.skip_window
        self.batch_size = batch_config.batch_size
        self.data_length = len(process_data)

    def Skim_Gram_batch(self):
        assert self.batch_size % self.num_skips == 0
        assert self.num_skips == 2 * self.skip_window
        batch = np.ndarray(shape=(self.batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
        span = 2 * self.skip_window + 1 # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(self.data[self.data_index])
            self.data_index = (self.data_index + 1) % self.data_length
        for i in range(self.batch_size // self.num_skips):
            target =self.skip_window
            targets_to_avoid = [self.skip_window]
            for j in range(self.num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)

                targets_to_avoid.append(target)
                batch[ i * self.num_skips + j ] = buffer[self.skip_window]
                labels[ i * self.num_skips + j, 0] = buffer[target]
            buffer.append(self.data[self.data_index])
            self.data_index = (self.data_index + 1 ) % self.data_length

        return batch, labels

    def CBOW_batch(self):
        assert self.batch_size % self.num_skips == 0
        assert self.num_skips == 2 * self.skip_window
        batch = np.ndarray(shape=(self.batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(self.batch_size/self.num_skips, 1), dtype=np.int32)
        span = 2 * self.skip_window + 1 # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(self.data[self.data_index])
            self.data_index = (self.data_index + 1) % self.data_length
        for i in range(self.batch_size // self.num_skips):
            target =self.skip_window
            targets_to_avoid = [self.skip_window]
            for j in range(self.num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)

                targets_to_avoid.append(target)
                batch[i * self.num_skips + j] = buffer[target]
            labels[i, 0] = buffer[self.skip_window]
            buffer.append(self.data[self.data_index])
            self.data_index = (self.data_index + 1) % self.data_length

        return batch, labels
