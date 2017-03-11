import unittest
from embeddings import batcher

from dataset import Dataset
import dataset
class MyTestCase(unittest.TestCase):
    def test_something(self):
        text = [0,1,2,3,4,5,6,7,8]
        batchconfig = batcher.BatcherConfig(8,2,1)
        batch = batcher.Batcher(text,batchconfig)
        batches, labels = batch.CBOW_batch()
        print batches
        print labels
        # self.assertEqual(batches, ['originated', 'originated', 'as', 'as', 'a', 'a', 'term', 'term'])


if __name__ == '__main__':
    unittest.main()
    text = dataset.read_data("text")
