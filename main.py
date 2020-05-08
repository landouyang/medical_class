# -*- coding: utf-8 -*-
import os
import argparse

import keras
from keras.layers import *
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from flyai.framework import FlyAI
from flyai.data_helper import DataHelper
from path import MODEL_PATH, DATA_PATH,FASTTEXT_TRAIN_DATA_PATH,FASTTEXT_TEST_DATA_PATH,FASTTEXT_MODEL_PATH,FASTTEXT_PRETRAIN_VEC_PATH
import pandas as pd
import numpy as np
from data_helper import load_dict, load_labeldict, get_batches, read_data, get_val_batch,covert2fasttext
from keras import Input, Model
import fasttext
from prediction import Prediction
from flyai.utils import remote_helper

'''
此项目为FlyAI2.0新版本框架，数据读取，评估方式与之前不同
2.0框架不再限制数据如何读取
样例代码仅供参考学习，可以自己修改实现逻辑。
模版项目下载支持 PyTorch、Tensorflow、Keras、MXNET、scikit-learn等机器学习框架
第一次使用请看项目中的：FlyAI2.0竞赛框架使用说明.html
使用FlyAI提供的预训练模型可查看：https://www.flyai.com/models
学习资料可查看文档中心：https://doc.flyai.com/
常见问题：https://doc.flyai.com/question.html
遇到问题不要着急，添加小姐姐微信，扫描项目里面的：FlyAI小助手二维码-小姐姐在线解答您的问题.png
'''

# 项目的超参，不使用可以删除
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
args = parser.parse_args()


class Main(FlyAI):

    '''
    项目中必须继承FlyAI类，否则线上运行会报错。
    '''
    def download_data(self):
        # 下载数据
        data_helper = DataHelper()
        data_helper.download_from_ids("MedicalClass")
        print('=*=数据下载完成=*=')

    def deal_with_data(self):
        '''
        处理数据，没有可不写。
        :return:
        '''
        # 加载数据
        self.data = pd.read_csv(os.path.join(DATA_PATH, 'MedicalClass/train.csv'))
        #self.train_data, self.valid_data = train_test_split(self.data, test_size=0.1, random_state=6, shuffle=True)
        #转换为fasttext 数据格式, label以"__label__"开头,如__label__equipment, 空格分开label和text
        #covert2fasttext(self.train_data,FASTTEXT_TRAIN_DATA_PATH)
        #covert2fasttext(self.valid_data, FASTTEXT_TEST_DATA_PATH)
        if not os.path.exists(FASTTEXT_TRAIN_DATA_PATH):
            covert2fasttext(self.data, FASTTEXT_TRAIN_DATA_PATH)
        print('=*=数据处理完成=*=')

    def train(self):
        # 必须使用该方法下载模型，然后加载
        #if not os.path.isfile(FASTTEXT_PRETRAIN_VEC_PATH):
        #   remote_helper.get_remote_date('https://www.flyai.com/m/cc.zh.300.vec')

        #model = fasttext.train_supervised(epoch=1,ws=3,dim=300,minn=1,maxn=5,minCount=1,wordNgrams=5,input=FASTTEXT_TRAIN_DATA_PATH,pretrainedVectors=FASTTEXT_PRETRAIN_VEC_PATH)
        #model = fasttext.train_supervised(epoch=25,dim=300,input=FASTTEXT_TRAIN_DATA_PATH, pretrainedVectors=FASTTEXT_PRETRAIN_VEC_PATH)
        model = fasttext.train_supervised(epoch=30, wordNgrams=3, lr=0.5, dim=300, input=FASTTEXT_TRAIN_DATA_PATH)
        model.save_model(FASTTEXT_MODEL_PATH)


if __name__ == '__main__':
    main = Main()
    main.download_data()
    main.deal_with_data()
    main.train()

    #pred = Prediction()
    #pred.load_model()
    #accu = pred.predict_file(FASTTEXT_TEST_DATA_PATH)
    #print("accu is{}".format(accu))
    # print(label)
    #pred.predict_3("癫痫犯病有什么现象", "朋友在三岁那年被确诊患有癫痫,这些年一直都在治疗,这几天姑姑姑父要出门办点事,就让我们家帮忙照顾表弟几天,我们从来都没有照顾过癫痫病人。")

    exit(0)