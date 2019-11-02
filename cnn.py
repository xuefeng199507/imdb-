
import time
from model.prepare_data import *
class cnn(object):
        def __init__(self, config):
            # configuration
            self.max_len = config["max_len"]
            # topic nums + 1
            self.num_classes = config["n_class"]
            self.vocab_size = config["vocab_size"]
            self.embedding_size = config["embedding_size"]
            self.filter_sizes = config["filter_sizes"]
            self.num_filters = config["num_filters"]
            self.l2_reg_lambda = config["l2_reg_lambda"]
            self.learning_rate = config["learning_rate"]

            # placeholder
            self.x = tf.placeholder(tf.int32, [None, self.max_len], name="input_x")
            self.label = tf.placeholder(tf.int32, [None, self.num_classes], name="input_y")
            self.dropoutKeepProb = tf.placeholder(tf.float32, name="keep_prob")

        def build_graph(self):
            print("building graph")
            l2Loss = tf.constant(0.0)

        # 词嵌入层
            with tf.name_scope("embedding"):
                self.W = tf.Variable(
                    tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                name="W")
                self.embedded_chars = tf.nn.embedding_lookup(self.W, self.x)  # batch_size * seq * embedding_size
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)  # expand dims for conv operation

            pooledOutputs = []
            # 有三种size的filter，3， 4， 5，textCNN是个多通道单层卷积的模型，可以看作三个单层的卷积模型的融合
            for i, filter_size in enumerate(self.filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # 卷积层，卷积核尺寸为filterSize * embeddingSize，卷积核的个数为numFilters
                    # 初始化权重矩阵和偏置
                    filterShape = [filter_size, self.embedding_size, 1, self.num_filters]

                    W = tf.Variable(tf.truncated_normal(filterShape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        )
                    # relu函数的非线性映射
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name = 'relu')
                    # 池化层，最大池化，池化是对卷积后的序列取一个最大值
                    pooled = tf.nn.max_pool( h,
                    ksize=[1, self.max_len - filter_size + 1, 1, 1],  # ksize shape: [batch, height, width, channels]
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                    pooledOutputs.append(pooled)  # 将三种size的filter的输出一起加入到列表中
            # 得到CNN网络的输出长度
            numFiltersTotal  = self.num_filters*len(self.filter_sizes)
            # 池化后的维度不变，按照最后的维度channel来concat
            self.hPool = tf.concat(pooledOutputs, 3)

                    # 摊平成二维的数据输入到全连接层
            self.hPoolFlat = tf.reshape(self.hPool, [-1, numFiltersTotal])

            # dropout
            with tf.name_scope("dropout"):
                self.hDrop = tf.nn.dropout(self.hPoolFlat, self.dropoutKeepProb)

            # 全连接层的输出
            with tf.name_scope("output"):
                outputW = tf.Variable(tf.truncated_normal([numFiltersTotal, self.num_classes], stddev=0.1), name="W")
                outputB = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="outputB")
                l2Loss += tf.nn.l2_loss(outputW)
                l2Loss += tf.nn.l2_loss(outputB)
                self.logits = tf.nn.xw_plus_b(self.hDrop, outputW, outputB, name="logits")
            with tf.name_scope('softmax'):
                self.result = tf.nn.softmax(logits=self.logits, dim=1, name='softmax')
            with tf.name_scope('loss'):
                self.real = tf.argmax(self.label, axis=1, name="real_label")
                self.predictions = tf.argmax(self.result, axis=1, name="predictions")

                losses = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.result))
                self.losses = losses + self.l2_reg_lambda * l2Loss
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.losses)

            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, self.real)
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            print("graph built successfully!")
if __name__ == '__main__':
    # load data
    x_train, y_train = load_data_and_labels("../data//train/pos_all.txt", "../data/train/neg_all.txt")
    x_test, y_test = load_data_and_labels("../data/test/test_pos.txt", "../data/test/test_neg.txt")
    # data preprocessing
    x_train, x_test, vocab, vocab_size = \
        data_preprocessing(x_train, x_test, max_len=120)
    print("train size: ", len(x_train))
    print("vocab size: ", vocab_size)

    # split dataset to test and dev
    x_test, x_dev, y_test, y_dev, dev_size, test_size = \
        split_dataset(x_test, y_test, 0.1)
    print("Validation Size: ", dev_size)

    config = {
        "max_len": 200,
        "vocab_size": vocab_size,
        "embedding_size": 32,
        "learning_rate": 1e-3,
        "l2_reg_lambda": 1e-3,
        "batch_size": 256,
        "n_class": 2,

        # random setting, may need fine-tune
        "filter_sizes": [1, 2, 3, 4, 5, 10, 20, 50, 100, 150],
        "num_filters": 8,
        "train_epoch": 10,
    }
    classifier = cnn(config)
    classifier.build_graph()

    # auto GPU growth, avoid occupy all GPU memory
    tf_config = tf.ConfigProto()
    sess = tf.Session(config=tf_config)

    sess.run(tf.global_variables_initializer())

    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        dev_batch = (x_dev, y_dev)
        start = time.time()
        print("Initialized! ")

        print("Start trainning")
        for e in range(20):

            epoch_start = time.time()
            t0 = time.time()
            print("Epoch %d start !" % (e + 1))
            for x_batch, y_batch in fill_feed_dict(x_train, y_train, config['batch_size']):

                fd = {classifier.x: x_batch, classifier.label: y_batch, classifier.dropoutKeepProb: 0.5}
                l, _, acc = sess.run([classifier.losses, classifier.optimizer, classifier.accuracy], feed_dict=fd)

            epoch_finish = time.time()
            print("Validation accuracy: ", sess.run([classifier.accuracy, classifier.losses], feed_dict={
                classifier.x: x_dev,
                classifier.label: y_dev,
                classifier.dropoutKeepProb: 1.0
                }))
            print("Epoch time: ", time.time() - epoch_start, "s")

        print("Training finished, time consumed : ", time.time() - start, " s")
        print("Start evaluating:  \n")
        cnt = 0
        test_acc = 0
        for x_batch, y_batch in fill_feed_dict(x_test, y_test, config['batch_size']):
            fd = {classifier.x: x_batch, classifier.label: y_batch, classifier.dropoutKeepProb: 1.0}
            acc = sess.run(classifier.accuracy, feed_dict=fd)
            test_acc += acc
            cnt += 1

        print("Test accuracy : %f %%" % (test_acc / cnt * 100))







