from model.multihead import *
from model.model_helper import *
import time
from model.prepare_data import *

class AttentionClassifier(object):
    def __init__(self, config):
        self.max_len = config["max_len"]
        self.hidden_size = config["hidden_size"]
        self.vocab_size = config["vocab_size"]
        self.embedding_size = config["embedding_size"]
        self.n_class = config["n_class"]
        self.learning_rate = config["learning_rate"]

        # placeholder
        self.x = tf.placeholder(tf.int32, [None, self.max_len])
        self.label = tf.placeholder(tf.int32, [None,self.n_class])
        self.keep_prob = tf.placeholder(tf.float32)

    def build_graph(self):
        print("building graph...")
        embeddings_var = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                                     trainable=True)
        batch_embedded = tf.nn.embedding_lookup(embeddings_var, self.x)
        print(batch_embedded.shape)
        # multi-head attention
        ma = multihead_attention(queries=batch_embedded, keys=batch_embedded)
        # FFN(x) = LN(x + point-wisely NN(x))
        outputs = feedforward(ma, [self.hidden_size, self.embedding_size])
        outputs = tf.reshape(outputs, [-1, self.max_len * self.embedding_size])
        logits = tf.layers.dense(outputs, units=self.n_class)

        self.losses = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.label))
        self.predictions = tf.argmax(tf.nn.softmax(logits), 1)
        self.real = tf.argmax(self.label, axis=1, name="real_label")
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.real)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        # optimization
        loss_to_minimize = self.losses
        tvars = tf.trainable_variables()
        gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        grads, global_norm = tf.clip_by_global_norm(gradients, 1.0)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step,
                                                       name='train_step')
        print("graph built successfully!")

if __name__ == '__main__':
        # load data
        x_train, y_train = load_data_and_labels("../data//train/pos_all.txt", "../data/train/neg_all.txt")
        x_test, y_test = load_data_and_labels("../data/test/test_pos.txt", "../data/test/test_neg.txt")
        # data preprocessing
        x_train, x_test, vocab, vocab_size = \
            data_preprocessing(x_train, x_test, max_len=32)
        print("train size: ", len(x_train))
        print("vocab size: ", vocab_size)

        # split dataset to test and dev
        x_test, x_dev, y_test, y_dev, dev_size, test_size = \
            split_dataset(x_test, y_test, 0.1)
        print("Validation Size: ", dev_size)

        config = {
            "max_len": 32,
            "hidden_size": 64,
            "vocab_size": vocab_size,
            "embedding_size": 32,
            "n_class": 2,
            "learning_rate": 1e-3,
            "batch_size": 32,
            "train_epoch": 20
        }
        classifier = AttentionClassifier(config)
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
                    fd = {classifier.x: x_batch, classifier.label: y_batch, classifier.keep_prob: 0.5}
                    l, _, acc = sess.run([classifier.losses, classifier.train_op, classifier.global_step], feed_dict=fd)


                epoch_finish = time.time()
                print("Validation accuracy: ", sess.run([classifier.accuracy, classifier.losses], feed_dict={
                    classifier.x: x_dev,
                    classifier.label: y_dev,
                    classifier.keep_prob: 1.0
                }))
                print("Epoch time: ", time.time() - epoch_start, "s")

            print("Training finished, time consumed : ", time.time() - start, " s")
            print("Start evaluating:  \n")
            cnt = 0
            test_acc = 0
            for x_batch, y_batch in fill_feed_dict(x_test, y_test, config['batch_size']):
                fd = {classifier.x: x_batch, classifier.label: y_batch, classifier.keep_prob: 1.0}
                acc = sess.run(classifier.accuracy, feed_dict=fd)
                test_acc += acc
                cnt += 1

            print("Test accuracy : %f %%" % (test_acc / cnt * 100))
