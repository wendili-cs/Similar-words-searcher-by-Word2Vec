#coding:utf-8
'''
TensorFlow 1.3
Python 3.6
By LiWenDi
'''
import collections
import math
import os
import random
import zipfile
import numpy as np
import urllib
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

TO_ContinueTRAIN = False #是否继续训练
TO_SAVE = False #是否保存模型
TO_JUDGE = False #是否进行测试训练完成的模型
TO_PIC = True #是否绘制特征空间图
VOCABULARY_SIZE = 50000 #选取最高频率的词汇数
BATCH_SIZE = 128
EMBEDDING_SIZE = 128 #单词转换为稠密向量的维度
SKIP_WINDOW = 1 #单词间最远可以联系的距离
NUM_SKIPS = 2 #每个目标单词提取的样本数
VALID_SIZE = 16 #抽取的验证单词数
VALID_WINDOW = 100 #从top多少进行抽取
VALID_EXAMPLES = np.random.choice(VALID_WINDOW, VALID_SIZE, replace = False)
NUM_SAMPLED = 64
NUM_STEPS = 100001 #训练次数
TOP_K = 8 #展示每个词的近义词的个数
PLOT_NUM = 100 #画出的示例点个数
OUTPUT_PATH = "Word2Vec_model/"
url = "http://mattmahoney.net/dc/"

if TO_JUDGE:
    TO_ContinueTRAIN = True

def maybe_download(filename, expected_bytes):
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print("文件找到且大小正确。 ——", filename)
    else:
        print(statinfo.st_size)
        raise Exception("认证失败：", filename, "。请在浏览器中下载。")
    return filename

filename = maybe_download("text8.zip", 31344016)

def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

words = read_data(filename)
print("文本数据大小： ", len(words))

def build_dataset(words):
    count = [["Unkown", -1]]
    count.extend(collections.Counter(words).most_common(VOCABULARY_SIZE - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)

del words
print("最高频的词有(+Unkown)： ", count[:5])
print("样本数据有： ", data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape = (batch_size), dtype = np.int32)
    labels = np.ndarray(shape = (batch_size, 1), dtype = np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen = span)

    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels

#--------------------------------------------小测试--------------------------------------------
batch, labels = generate_batch(batch_size = 8, num_skips = 2, skip_window = 1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]], "->", labels[i, 0], reverse_dictionary[labels[i, 0]])
#--------------------------------------------小测试--------------------------------------------
graph = tf.Graph()
with graph.as_default():

    train_inputs = tf.placeholder(tf.int32, shape = [BATCH_SIZE])
    train_labels = tf.placeholder(tf.int32, shape = [BATCH_SIZE, 1])
    valid_dataset = tf.constant(VALID_EXAMPLES, dtype = tf.int32)

    with tf.device('/cpu:0'):
        embeddings = tf.Variable(tf.random_uniform([VOCABULARY_SIZE, EMBEDDING_SIZE], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        nce_weights = tf.Variable(tf.truncated_normal([VOCABULARY_SIZE, EMBEDDING_SIZE], stddev = 1.0 / math.sqrt(EMBEDDING_SIZE)))
        nce_biases = tf.Variable(tf.zeros([VOCABULARY_SIZE]))

    loss = tf.reduce_mean(tf.nn.nce_loss(weights = nce_weights, biases = nce_biases, labels = train_labels,
                                         inputs = embed, num_sampled = NUM_SAMPLED, num_classes = VOCABULARY_SIZE))

    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims = True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b = True) #转置了后者



with tf.Session(graph = graph) as session:
    saver = tf.train.Saver()
    if TO_ContinueTRAIN:
        saver.restore(session, OUTPUT_PATH)
        print("加载已训练模型完成")
    else:
        init = tf.global_variables_initializer()
        init.run()
        print("初始化完成")
    if not TO_JUDGE:
        average_loss = 0
        for step in range(NUM_STEPS):
            bacth_inputs, batch_labels = generate_batch(BATCH_SIZE, NUM_SKIPS, SKIP_WINDOW)
            feed_dict = {train_inputs: bacth_inputs, train_labels:batch_labels}

            _, loss_val = session.run([optimizer, loss], feed_dict = feed_dict)
            average_loss += loss_val

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                print("第", step, "步，此时的平均loss为：", average_loss)
                print("---------------------------------------------------------------------------")
                average_loss = 0

            if step % 10000 == 0:
                sim = similarity.eval()
                for i in range(VALID_SIZE):
                    valid_word = reverse_dictionary[VALID_EXAMPLES[i]]
                    nearest = (-sim[i, :]).argsort()[1:TOP_K+1]
                    log_str = "%s 的近似特征词: " % valid_word
                    for k in range(TOP_K):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = "%s %s," % (log_str, close_word)
                    print(log_str)
                print("---------------------------------------------------------------------------")
                final_embeddings = normalized_embeddings.eval()
        if TO_SAVE:
            saver.save(session, OUTPUT_PATH)
            print("模型已储存")
    else:
        word_search = input("请输入你想查询的词语( -1 表示结束程序)：\n")
        while(word_search != "-1"):
            if word_search in dictionary:
                target_in_dict = tf.nn.embedding_lookup(normalized_embeddings, dictionary[word_search])
                target_in_dict = tf.reshape(target_in_dict, [1, 128])
                simi = tf.matmul(target_in_dict, normalized_embeddings, transpose_b = True)
                nearest = (-session.run(simi[0,:])).argsort()[1:TOP_K + 1]
                log_str = "%s 的近似特征词有: " % word_search
                for k in range(TOP_K):
                            close_word = reverse_dictionary[nearest[k]]
                            log_str = "%s %s," % (log_str, close_word)
                print(log_str)
                word_search = input("请输入你想查询的词语( -1 表示结束程序 )：\n")
            else:
                word_search = input("查无此词！请重新输入\n")
        print("程序已结束。")

#可视化函数
def plot_with_labels(low_dim_embs, labels, filename = 'word2vec.png'):
    assert low_dim_embs.shape[0] >= len(labels), "标签多于了嵌入量！"
    plt.figure(figsize = (18, 18))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy = (x, y), xytext = (5, 2), textcoords = "offset points", ha = "right", va = "bottom")
    plt.savefig(filename)
    print("绘制图像完成！")

if TO_PIC and not TO_JUDGE:
    tsne = TSNE(perplexity = 30, n_components = 2, init = "pca", n_iter = 5000)
    low_dim_embs = tsne.fit_transform(final_embeddings[:PLOT_NUM, :])
    labels = [reverse_dictionary[i] for i in range(PLOT_NUM)]
    plot_with_labels(low_dim_embs, labels)