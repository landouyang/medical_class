# -*- coding: utf-8 -*
import sys
import os

# 训练数据的路径
DATA_PATH = os.path.join(sys.path[0], 'data', 'input')

#保存FastText train data文件
FASTTEXT_TRAIN_DATA_PATH = os.path.join(sys.path[0], 'data', 'input','fasttext.train.txt')

#保存FastText test data文件
FASTTEXT_TEST_DATA_PATH = os.path.join(sys.path[0], 'data', 'input','fasttext.test.txt')

#保存FastText pretrained vec文件
FASTTEXT_PRETRAIN_VEC_PATH = os.path.join(sys.path[0], 'data', 'input','model','cc.zh.300.vec')

#保存FastText pretrained vec文件
#FASTTEXT_PRETRAIN_VEC_PATH = os.path.join(sys.path[0], 'data', 'input','cc.zh.300.vec')


#保存FastText pretrained bin文件
FASTTEXT_PRETRAIN_BIN_PATH = os.path.join(sys.path[0], 'data', 'input','cc.zh.300.bin')


#保存FastText train后的model文件
FASTTEXT_MODEL_PATH = os.path.join(sys.path[0], 'data', 'output','medical_class.bin')

# 模型保存的路径
MODEL_PATH = os.path.join(sys.path[0], 'data', 'output', 'model')
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)