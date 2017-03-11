__author__ = 'pandaczm'

import numpy as np
import string
import collections
import zipfile
import tensorflow as tf



class Dataset:


    @staticmethod
    def create_from_text(text, num_gram):
        """
        :param text - initial text to process
        :return:
         bigram2id - mapping bigram value to bigram id(index)
         text_from_grams - initial text where letters is substituted by appropriate bigram id
         bigrams - bigrams array
        """
        grams2id = {}
        text_from_grams = []

        for char in text:
            if char not in (string.ascii_lowercase + " "):
                text = text.replace(char, " ")
        if num_gram == 0:
            text_grams = text.split()
            count = [['UNK', -1]]
            count.extend(collections.Counter(text_grams).most_common(50000 - 1))
            grams2id = dict()
            for word, _ in count:
                grams2id[word] = len(grams2id)

            unk_count = 0
            for word in text:
                if word in grams2id:
                    index = grams2id[word]
                else:
                    index = 0  # grams2id['UNK']
                    unk_count = unk_count + 1
                text_from_grams.append(index)
            count[0][1] = unk_count
            print len(grams2id)
            id2gram = dict(zip(grams2id.values(), grams2id.keys()))


        else:
            text_grams = [text[i:i + num_gram] for i in range(0, len(text)/num_gram * num_gram, num_gram)]
            for word in text_grams:
                if word not in grams2id:
                    num = len(grams2id)
                    grams2id[word] = num
                text_from_grams.append(num)
            # print len(set(grams2id.values())), len(grams2id.keys())
            # print grams2id.values()


            id2gram = dict(zip(grams2id.values(), grams2id.keys()))
        # print len(grams2id), len(id2gram)
        assert len(grams2id) == len(id2gram)

        return Dataset(grams2id,text_from_grams,id2gram)



        # cursor = 0
        # while cursor < len(text) - 1:
        #     bigram = text[cursor] + text[cursor + 1]
        #     if bigram not in bigrams2id:
        #         bigram_id = len(bigrams2id)
        #         bigrams2id[bigram] = bigram_id
        #         bigrams.append(bigram)
        #         text_from_grams.append(bigram_id)
        #     else:
        #         text_from_grams.append(bigrams2id[bigram])
        #
        #     cursor += 2
        #
        # assert len(bigrams) == len(bigrams2id)
        # return Dataset(bigrams2id, text_from_grams, bigrams)

    def __init__(self, grams2id, text_from_grams, id2gram):
        self.grams2id = grams2id
        self.text_from_grams = text_from_grams
        self.id2gram = id2gram
        self.vocabulary_size = len(grams2id)
        self.text_from_bigrams_len = len(self.text_from_grams)

    def train_validation_split(self, validation_size):
        """
        :param validation_size:
        :return: train and validation datasets
        """
        assert validation_size >= self.get_grams_len()

        validation_dataset = Dataset(self.grams2id, self.text_from_grams[-validation_size:], self.id2gram)
        train_dataset = Dataset(self.grams2id, self.text_from_grams[:-validation_size], self.id2gram)
        return train_dataset, validation_dataset

    def get_gram_from_id(self, id_num):
        return self.id2gram[id_num]

    def get_id_from_gram(self, gram):
        return self.grams2id[gram]

    def get_grams_len(self):
        return len(self.id2gram)


class DatasetWithEmbeddings:
    def __init__(self, dataset, embeddings):
        self.processed_data = dataset
        self.embeddings = embeddings

    def get_embedding_in_position(self, position):
        gram_id = self.processed_data.text_from_grams[position]
        return self.embeddings[gram_id]

    def get_gram_id_in_position(self, position):
        return self.processed_data.text_from_grams[position]

    def get_data_len(self):
        return len(self.processed_data.text_from_bigrams)

    def get_embedding_size(self):
        return self.embeddings.shape[1]

    def get_gram_from_embedding(self, embedding):
        embedding_np = np.array(embedding)
        current_dist = np.inf
        current_closest = 0
        for i in range(len(self.embeddings)):
            dist = np.linalg.norm(embedding_np - self.embeddings[i])
            if dist < current_dist:
                current_closest = i
                current_dist = dist
        return self.processed_data.get_bigram(current_closest)

    def get_embedding_for_gram(self, gram):
        id = self.processed_data.get_id_from_gram(gram)
        return self.embeddings[id]

    def get_gram(self, id):
        return self.processed_data.get_gram_from_id(id)

    def get_grams_len(self):
        return self.processed_data.get_grams_len()

def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        name = f.namelist()[0]
        data = tf.compat.as_str(f.read(name))
    return data

