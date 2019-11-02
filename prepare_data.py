import numpy as np
import re
import tensorflow as tf
from sklearn.utils import shuffle
import os
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
stop = set(stopwords.words('english')) #
#print(stop)
def remove_stop(data):

    after_remove = list()
    length = len(data) # 获取数据集的大小 一般是一个 ndarray
    total_words = 0 # 用于计算平均长度 以方便后面截断长度的设定
    for i in range(length): # 依次处理文本
        clean_sentence = list()
        for word in data[i].split(): # 将文本 spilit 成一个个单词
            if word not in stop:
                clean_sentence.append(word)
        total_words += len(clean_sentence)
        after_remove.append(" ".join(clean_sentence))
    print("Average length:", total_words/length)
    return np.asarray(after_remove)


def remove_stop_word(article):
    new_article = []
    for word in article:
        if word not in stopwords:
            new_article.append(word)
    return new_article

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()#返回字符串去除首尾空格后的小写字符串


def load_data_and_labels(positive_data_file, negative_data_file, sample_ratio=1):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())#读取每行
    positive_examples = [s.strip() for s in positive_examples]#去掉每行首尾空白
    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)#拼接
    return [x, y]

filename = r'C:\Users\Administrator\Desktop\数据集\glove.42B.300d.txt'
# def loadGloVe(filename):
#     vocab = []
#     embd = []
#     file = open(filename, 'r',encoding = 'UTF-8')
#     for line in file.readlines(): # 读取 txt 的每一行
#         row = line.strip().split(' ')
#         vocab.append(row[0])
#         embd.append(row[1:])
#     print('Loaded GloVe!')
#     file.close()
#     return vocab, embd
# vocab, embd = loadGloVe(filename)
# vocab_size = len(vocab) # 词表的大小
# embedding_dim = len(embd[0]) # embedding 的维度
# print("Vocab size : ", vocab_size)
# print("Embedding dimensions : ", embedding_dim)

def data_preprocessing(x_train,x_test,max_len):
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_len)
    x_transform_train = vocab_processor.fit_transform(x_train)
    x_transform_test = vocab_processor.transform(x_test)#上面三行 将文本变成向量
    vocab = vocab_processor.vocabulary_#计数
    vocab_size = len(vocab)
    x_train_list = list(x_transform_train)
    x_test_list = list(x_transform_test)
    x_train = np.array(x_train_list)
    x_test = np.array(x_test_list)#将列表转换为array

    return x_train, x_test, vocab, vocab_size


def split_dataset(x_test, y_test, dev_ratio):#划分验证集与测试集
    """split test dataset to test and dev set with ratio """
    test_size = len(x_test)
    print(test_size)
    dev_size = (int)(test_size * dev_ratio)
    print(dev_size)
    x_dev = x_test[:dev_size]
    x_test = x_test[dev_size:]
    y_dev = y_test[:dev_size]
    y_test = y_test[dev_size:]
    return x_test, x_dev, y_test, y_dev, dev_size, test_size - dev_size
def fill_feed_dict(data_X, data_Y, batch_size):
    """Generator to yield batches"""
    # Shuffle data first. 随机排序数据
    shuffled_X, shuffled_Y = shuffle(data_X, data_Y)
    # print("before shuffle: ", data_Y[:10])
    # print(data_X.shape[0])
    # perm = np.random.permutation(data_X.shape[0])
    # data_X = data_X[perm]
    # shuffled_Y = data_Y[perm]
    # print("after shuffle: ", shuffled_Y[:10])
    for idx in range(data_X.shape[0] // batch_size):
        x_batch = shuffled_X[batch_size * idx: batch_size * (idx + 1)]
        y_batch = shuffled_Y[batch_size * idx: batch_size * (idx + 1)]
        yield x_batch, y_batch
def data_preprocessing_with_dict(train, test, max_len):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='<UNK>')
    tokenizer.fit_on_texts(train)
    train_idx = tokenizer.texts_to_sequences(train)
    test_idx = tokenizer.texts_to_sequences(test)
    train_padded = pad_sequences(train_idx, maxlen=max_len, padding='post', truncating='post')
    test_padded = pad_sequences(test_idx, maxlen=max_len, padding='post', truncating='post')
    # vocab size = len(word_docs) + 2  (<UNK>, <PAD>)
    return train_padded, test_padded, tokenizer.word_docs, tokenizer.word_index, len(tokenizer.word_docs) + 2