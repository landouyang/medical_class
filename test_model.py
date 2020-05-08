from fasttext import load_model
from path import FASTTEXT_MODEL_PATH,FASTTEXT_PRETRAIN_BIN_PATH

f = load_model(FASTTEXT_MODEL_PATH)
word = "月经"
vec = f.get_word_vector(word)
ns = f.get_nearest_neighbors(word,k=5)
print("The vector of word: {} is {}".format(word,vec))
print("nearest_neighbors is {}".format(ns))

labels = f.get_labels()
print(labels)