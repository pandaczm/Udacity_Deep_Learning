__author__ = 'pandaczm'

import numpy as np

class BatchConfig(object):
    def __init__(self, batch_size, num_unrollings):
        self.batch_size = batch_size
        self.num_unrollings = num_unrollings

class BatchGenerator:
  def __init__(self, batchconfig, datasetwithembeddings):
    self.dataset_embeddings = datasetwithembeddings
    self.text = datasetwithembeddings.processed_data.grams_id_from_text
    self.text_size = datasetwithembeddings.processed_data.total_grams_len
    self.batch_size = batchconfig.batch_size
    self.num_unrollings = batchconfig.num_unrollings
    segment = self.text_size // self.batch_size
    self._cursor = [offset * segment for offset in range(self.batch_size)]

    self.embeddings_size = self.dataset_embeddings.embeddings_size
    self._last_batch = self._next_batch()

  def _next_batch(self):
    """Generate a single batch from the current cursor position in the data."""
    batch = np.zeros(shape=(self.batch_size), dtype=np.int)
    label = np.zeros(shape=(self.batch_size,1), dtype = np.int)
    for b in range(self.batch_size):
        batch[b] = self.text[self._cursor[b]]
        label[b,0] = self.text[self._cursor[b]]
        self._cursor[b] = (self._cursor[b] + 1) % self.text_size
    return batch, label

  def next(self):
    """Generate the next array of batches from the data. The array consists of
    the last batch of the previous array, followed by num_unrollings new ones.
    """
    batches = [self._last_batch[0]]
    labels = [self._last_batch[1]]
    for step in range(self.num_unrollings):
        a, b = self._next_batch()
        batches.append(a)
        labels.append(b)
    self._last_batch = [batches[-1], labels[-1]]
    return batches


if __name__ == "__main__":
    c = BatchConfig(2,1)