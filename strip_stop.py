import re
from types import MethodType, FunctionType

import jieba


def clean_txt(raw):
    fil = re.compile(r"[^0-9a-zA-Z\u4e00-\u9fa5]+")
    return fil.sub(' ', raw)


def seg(sentence, sw, apply=None):
    if isinstance(apply, FunctionType) or isinstance(apply, MethodType):
        sentence = apply(sentence)
    return ' '.join([i for i in jieba.cut(sentence) if i.strip() and i not in sw])
    #return ' '.join([i for i in sentence if i.strip() and i not in sw])


def stop_words():
    with open('stopwords.txt', 'r', encoding='utf-8') as swf:
        return [line.strip() for line in swf]

def process(sentence):
    tokens = seg(sentence,stop_words(),apply=clean_txt)
    return tokens

# 对某个sentence进行处理：
#content = '甲状腺瘤患者治疗要多少价钱?,"从上个月开始我就感觉身体无力,常的爱出虚汗食欲也不太致身体非常的瘦,去医院检查,生说是甲状腺腺瘤,问甲状腺瘤患者治疗要多少价钱?"'
#res = seg(content.lower().replace('\n', ''), stop_words(), apply=clean_txt)
#res = process(content)
#print(res)