#! -*- coding:utf-8 -*-
# 数据集：IFLYTEK' 长文本分类 (https://github.com/CLUEbenchmark/CLUE)
# 博客：https://kexue.fm/archives/7234

import argparse
import json
import numpy as np
from sklearn.model_selection import KFold
from bert4keras.backend import set_gelu
from bert4keras.backend import keras, search_layer, K
from bert4keras.tokenizer import Tokenizer
from bert4keras.bert import build_bert_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from keras.layers import Lambda, Dense
from tqdm import tqdm
from keras.layers import *

import pandas as pd

#set_gelu('tanh')  # 切换gelu版本

## HuaWei NeTha
#config_path = 'NEZHA/bert_config.json'
#checkpoint_path = 'NEZHA/model.ckpt-900000'
#dict_path = 'NEZHA/vocab.txt'


def load_data(filename):
    D = pd.read_csv(filename).values.tolist()
    return D





class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=True):
        idxs = list(range(len(self.data)))
        if random:
            np.random.shuffle(idxs)
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for i in idxs:
            #print(self.data[i])
            _,_, text1, text2, label = self.data[i]
#             print(text1, text2, label)
            token_ids, segment_ids = tokenizer.encode(text1, text2, max_length=args.maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []



def adversarial_training(model, embedding_name, epsilon=1):
    """给模型添加对抗训练
    其中model是需要添加对抗训练的keras模型，embedding_name
    则是model里边Embedding层的名字。要在模型compile之后使用。
    """
    if model.train_function is None:  # 如果还没有训练函数
        model._make_train_function()  # 手动make
    old_train_function = model.train_function  # 备份旧的训练函数

    # 查找Embedding层
    for output in model.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break
    if embedding_layer is None:
        raise Exception('Embedding layer not found')

    # 求Embedding梯度
    embeddings = embedding_layer.embeddings  # Embedding矩阵
    gradients = K.gradients(model.total_loss, [embeddings])  # Embedding梯度
    gradients = K.zeros_like(embeddings) + gradients[0]  # 转为dense tensor

    # 封装为函数
    inputs = (model._feed_inputs +
              model._feed_targets +
              model._feed_sample_weights)  # 所有输入层
    embedding_gradients = K.function(
        inputs=inputs,
        outputs=[gradients],
        name='embedding_gradients',
    )  # 封装为函数

    def train_function(inputs):  # 重新定义训练函数
        grads = embedding_gradients(inputs)[0]  # Embedding梯度
        delta = epsilon * grads / (np.sqrt((grads**2).sum()) + 1e-8)  # 计算扰动
        K.set_value(embeddings, K.eval(embeddings) + delta)  # 注入扰动
        outputs = old_train_function(inputs)  # 梯度下降
        K.set_value(embeddings, K.eval(embeddings) - delta)  # 删除扰动
        return outputs

    model.train_function = train_function  # 覆盖原训练函数


def make_model(config_path, checkpoint_path, prefix):

    if prefix == 'BERT':
        bert = build_bert_model(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            with_pool=True,
            return_keras_model=False,
        )
    if prefix == 'NEZHA':
        bert = build_bert_model(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            model='nezha',
            with_pool=True,
            return_keras_model=False,
        )   
                          

    output = Dropout(rate=0.01)(bert.model.output)
    ## 加了adversarial 层后，可以考虑更稳定些
    #output = Lambda(lambda x: x[:, 0])(bert.model.output)

    output = Dense(units=2,
                activation='softmax',
                kernel_initializer=bert.initializer)(output)

    model = keras.models.Model(bert.model.input, output)
    # model.summary()
    
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(args.lr),
        metrics=['accuracy'],
    )
    # 写好函数后，启用对抗训练只需要一行代码
    adversarial_training(model, 'Embedding-Token', args.alpha)

    return model


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    def __init__(self, generator, prefix, best_val_acc=0., num=0):
        self.best_val_acc = best_val_acc
        self.num = int(num)
        self.generator = generator
        self.prefix = prefix

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(self.generator)
        if val_acc > self.best_val_acc:
            print('Model ckpt store!')
            self.best_val_acc = val_acc
            model.save_weights('{0}_best_{1}_model.weights'.format(args.prefix, self.num))
        #test_acc = evaluate(valid_generator)
        test_acc = val_acc
        print(u'test_acc: %.5f, best_test_acc: %.5f, val_acc: %.5f\n'
              % (val_acc, self.best_val_acc, test_acc))


def train(model, train_data, valid_data, batch_size, turn):
    train_generator = data_generator(train_data, batch_size)
    valid_generator = data_generator(valid_data, batch_size)

    evaluator = Evaluator(generator=valid_generator, prefix=args.prefix, num=turn)
    model.fit_generator(train_generator.forfit(),
                        steps_per_epoch=len(train_generator),
                        epochs=args.epochs,
                        callbacks=[evaluator])
    model.load_weights('{0}_best_{1}_model.weights'.format(args.prefix, turn))
    best_score = evaluate(valid_generator)
    with open('{}_record_acc.txt'.format(args.prefix), 'a+') as f:
        f.write('Turn {0} Best acc {1:.4f}\n'.format(turn, best_score))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='adver',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--prefix', type=str, default='BERT')
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--maxlen', type=int, default=128)
    parser.add_argument('--alpha', type=float, default=0.5)
    args = parser.parse_args()
    print(args)

    config_path = args.prefix+'/bert_config.json'
    checkpoint_path = args.prefix+'/bert_model.ckpt'

    dict_path = args.prefix + '/vocab.txt'
    tokenizer = Tokenizer(dict_path, do_lower_case=True)
    # model = make_model(config_path, checkpoint_path)
    

    print('Start Training...') 
    all_data = load_data('./data/train.csv')
    random_order = list(range(len(all_data)))
    # np.random.shuffle(random_order)

    kf = KFold(n_splits=5, shuffle=True)
    turn = 1
    for train_index, test_index in kf.split(random_order):
        train_data = [all_data[i] for i in train_index]
        test_data= [all_data[j] for j in test_index]
        print('*****************Turn {}**********************'.format(turn))
        model = make_model(config_path, checkpoint_path, args.prefix)
        train(model, train_data, test_data, args.bs, turn)
        turn = turn + 1
