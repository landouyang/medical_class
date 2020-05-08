# -*- coding: utf-8 -*
import os
import numpy as np
from flyai.framework import FlyAI
from path import MODEL_PATH, DATA_PATH
from data_helper import pred_process, load_dict, load_labeldict,read_data
from path import FASTTEXT_MODEL_PATH,FASTTEXT_TEST_DATA_PATH
from fasttext import load_model
import pandas as pd
import csv
from strip_stop import process


class Prediction(FlyAI):

    def load_model(self):
        '''
        模型初始化，必须在构造方法中加载模型
        '''
        self.model = load_model(FASTTEXT_MODEL_PATH)
        return  self.model
        #print(self.model.get_labels())

    def predict(self, title, text):
        '''
        模型预测返回结果
        :param input: 评估传入样例 {"title": "心率为72bpm是正常的吗", "text": "最近不知道怎么回事总是感觉心脏不舒服..."}
        :return: 模型预测成功中户 {"label": "心血管科"}
        '''

        if title == "":
            text_line = text.replace(" ", "")
        elif text == "":
            text_line = title.replace(" ", "")
        else:
            text_line = title.replace(" ", "") + "," + text.replace(" ", "")

        text_line = text_line.replace("\r", "")
        text_line = text_line.replace("\n", "")
        tokens = process(text_line)
        label_tuple = self.model.predict(tokens)

        #print(label_tuple)
        label_pre_text = label_tuple[0][0]
        #print("label is {}"+label_pre_text)
        label = label_pre_text[label_pre_text.index("__label__")+9:]
        #print(label +": "+tokens[:51])
        return {'label': label}

    def predict_file(self,test_file):
        with open(test_file,"r",encoding="UTF-8") as tf:
            lines = tf.readlines()
            right_count = 0
            for line in lines:
                label_pre = line[0:line.index(" ")]
                text = line[line.index(" "):]
                text = text.replace("\n", "")
                label_pre_text = self.predict("",text)
                if label_pre_text["label"] == label_pre[9:]:
                    right_count +=1
            accu = right_count/len(lines)*100
        return accu

    def predict_3(self, title, text):
        '''
        模型预测返回结果
        :param input: 评估传入样例 {"title": "心率为72bpm是正常的吗", "text": "最近不知道怎么回事总是感觉心脏不舒服..."}
        :return: 模型预测成功中户 {"label": "心血管科"}
        '''
        label_tuple = self.model.predict(title+","+text,k=3)

        print(label_tuple)

    def fast_text_test(self):
        '''
        模型预测返回结果
        :param input: 评估传入样例 {"title": "心率为72bpm是正常的吗", "text": "最近不知道怎么回事总是感觉心脏不舒服..."}
        :return: 模型预测成功中户 {"label": "心血管科"}
        '''
        return self.model.test(FASTTEXT_TEST_DATA_PATH)


# if __name__ == '__main__':
#     pred = Prediction()
#
#     pred.load_model()
#
#     result = pred.fast_text_test()
#     print("the test result is {}".format(result))
    # label = pred.predict("癫痫犯病有什么现象","朋友在三岁那年被确诊患有癫痫,这些年一直都在治疗,这几天姑姑姑父要出门办点事,就让我们家帮忙照顾表弟几天,我们从来都没有照顾过癫痫病人。")
    # print(label)
#     accu = pred.predict_file(FASTTEXT_TEST_DATA_PATH)
#     print("the accu result is {}".format(accu))
#
#     #result = pred.fast_text_test()
#     #print("the test result is {}".format(result))
#
#     exit(0)