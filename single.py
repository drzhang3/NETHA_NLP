# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 19:27:50 2020

@author: kasy
"""

from keras.layers import *

from bert4keras.backend import keras, set_gelu
from bert4keras.bert import build_bert_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.tokenizer import Tokenizer
import pandas as pd
import numpy as np


#set_gelu('tanh')  # 切换gelu版本

prefix = 'HUAWEI'


def load_data(filename):
    D = pd.read_csv(filename).values.tolist()
    return D


test_data = load_data('/tcdata/test.csv')


batch_size = 1


class test_data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        idxs = list(range(len(self.data)))
        if random:
            np.random.shuffle(idxs)
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for i in idxs:
            #print(self.data[i])
            idx_i,_, text1, text2, label = self.data[i]
#             print(text1, text2, label)
            token_ids, segment_ids = tokenizer.encode(text1, text2, max_length=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            #batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                #batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], idx_i
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []
                



def eval_submission(data):
    #total, right = 0., 0.
    pred_list = []
    idx_list = []
    for x_true, y_idx in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        #y_true = y_true[:, 0]
        pred_list.append(int(y_pred))
        idx_list.append(y_idx)
        #print(y_idx, y_pred)
    pred_list = np.asarray(pred_list)
    return idx_list, pred_list

test_generator = test_data_generator(test_data, 1)
print('{0}_best_model.weights'.format(prefix))


config_path = 'HUAWEI/bert_config.json'
checkpoint_path = 'HUAWEI/bert_model.ckpt'
dict_path = 'HUAWEI/vocab.txt'          

tokenizer = Tokenizer(dict_path, do_lower_case=True)



bert = build_bert_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    with_pool=True,
    return_keras_model=False,
)    

output = Dropout(rate=0.01)(bert.model.output)
output = Dense(units=2,
               activation='softmax',
               kernel_initializer=bert.initializer)(bert.model.output)

model = keras.models.Model(bert.model.input, output)
model.summary()
#==============load second checkpoints=====================

model.load_weights('{0}_best_1_model.weights'.format(prefix))
idxs, preds = eval_submission(test_generator)


print('single')

submission = pd.DataFrame({'id': idxs,
                         'label': preds})

submission_file = 'result.csv'
submission.to_csv(submission_file, index=False)


