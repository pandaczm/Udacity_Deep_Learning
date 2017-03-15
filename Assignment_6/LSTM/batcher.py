__author__ = 'pandaczm'

import numpy as np

class BatchConfig(object):
    def __init__(self, batch_size, num_unrollings):
        self.batch_size = batch_size
        self.num_unrollings = num_unrollings

class BatchGenerator:
  def __init__(self, batchconfig, datasetwithembeddings):
    self._dataset_embeddings = datasetwithembeddings
    self._text = datasetwithembeddings.processed_data.grams_id_from_text
    self._text_size = datasetwithembeddings.processed_data.total_grams_len
    self._batch_size = batchconfig.batch_size
    self._num_unrollings = batchconfig.num_unrollings
    segment = self._text_size // self._batch_size
    self._cursor = [ offset * segment for offset in range(self._batch_size)]

    self.embeddings_size = self._dataset_embeddings.embeddings_size
    self._last_batch = self._next_batch()

  def _next_batch(self):
    """Generate a single batch from the current cursor position in the data."""
    batch = np.zeros(shape=(self._batch_size, self.embeddings_size), dtype=np.float)
    for b in range(self._batch_size):
        batch[b] = self._dataset_embeddings.get_embedding_in_id(self._text[self._cursor[b]])
        self._cursor[b] = (self._cursor[b] + 1) % self._text_size
    return batch

  def next(self):
    """Generate the next array of batches from the data. The array consists of
    the last batch of the previous array, followed by num_unrollings new ones.
    """
    batches = [self._last_batch]
    for step in range(self._num_unrollings):
        batches.append(self._next_batch())
    self._last_batch = batches[-1]
    return batches


if __name__ == "__main__":
    c = BatchConfig(2,1)