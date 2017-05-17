import pandas as pd
import numpy as np
import tensorflow as tf
from collections import Counter
from sklearn.datasets import fetch_20newsgroups

#категории для классификации
categories = ["comp.graphics", "sci.space", "rec.sport.baseball"]

#набор тренировочных данных
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)

#набор данных для тестирования
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

#концвертация слов в индексы


def get_word_2_index(vocab):
    word2index = {}
    for i, word in enumerate(vocab):
        word2index[word.lower()] = i

    return word2index


def get_batch(df, i, batch_size):
    batches = []
    results = []
    texts = df.data[i * batch_size:i * batch_size + batch_size]
    categories = df.target[i * batch_size:i * batch_size + batch_size]
    for text in texts:
        layer = np.zeros(total_words, dtype=float)
        for word in text.split(' '):
            layer[word2index[word.lower()]] += 1

        batches.append(layer)

    for category in categories:
        y = np.zeros((3), dtype=float)
        if category == 0:
            y[0] = 1.
        elif category == 1:
            y[1] = 1.
        else:
            y[2] = 1.
        results.append(y)

    return np.array(batches), np.array(results)


def multilayer_perceptron(input_tensor, weights, biases):
    layer_1_multiplication = tf.matmul(input_tensor, weights['h1'])
    layer_1_addition = tf.add(layer_1_multiplication, biases['b1'])
    layer_1 = tf.nn.relu(layer_1_addition)

    # Скрытый слой с RELU активацией ( f(x) = max (x,0) )
    layer_2_multiplication = tf.matmul(layer_1, weights['h2'])
    layer_2_addition = tf.add(layer_2_multiplication, biases['b2'])
    layer_2 = tf.nn.relu(layer_2_addition)

    # Выходной слой с линейной активацией
    out_layer_multiplication = tf.matmul(layer_2, weights['out'])
    out_layer_addition = out_layer_multiplication + biases['out']

    return out_layer_addition

vocab = Counter()

#добавление в коллекцию слов из тренировочной и тестовой выборок
for text in newsgroups_train.data:
    for word in text.split(' '):
        vocab[word.lower()] += 1

for text in newsgroups_test.data:
    for word in text.split(' '):
        vocab[word.lower()] += 1

total_words = len(vocab)
word2index = get_word_2_index(vocab)


def get_batch(df, i, batch_size):
    batches = []
    results = []
    texts = df.data[i * batch_size:i * batch_size + batch_size]
    categories = df.target[i * batch_size:i * batch_size + batch_size]
    for text in texts:
        layer = np.zeros(total_words, dtype=float)
        for word in text.split(' '):
            layer[word2index[word.lower()]] += 1

        batches.append(layer)

    for category in categories:
        y = np.zeros((3), dtype=float)
        if category == 0:
            y[0] = 1.
        elif category == 1:
            y[1] = 1.
        else:
            y[2] = 1.
        results.append(y)

    return np.array(batches), np.array(results)

#Параметры нейронной сети
learning_rate = 0.01
training_epochs = 2
batch_size = 150
display_step = 1

#Параметры нейронной сети
n_hidden_1 = 100      # количество нейронов в 1 скрытом слое
n_hidden_2 = 25       # количество нейронов во 2 скрытом слое
n_input = total_words # количество слов в словаре
n_classes = 3         # количество категорий классификации

#Входной и выходной массив данных
input_tensor = tf.placeholder(tf.float32, [None, n_input], name="input")
output_tensor = tf.placeholder(tf.float32, [None, n_classes], name="output")

#генерация весов сети с нормальным распределением
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
#генерация смещений сети с нормальным распределением
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# создание модели сети
prediction = multilayer_perceptron(input_tensor, weights, biases)

# Определение ошибок с использованием метода перекрестной энтропии
# Среднее значение ошибок
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=output_tensor))
# Обновление весов в процессе обучения с учетом минимизации потерь и скорости обучения
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Инициализация переменых в tensorflow
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Тренировочный цикл
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(newsgroups_train.data)/batch_size)
        # Цикл по всем батчам
        for i in range(total_batch):
            batch_x, batch_y = get_batch(newsgroups_train, i, batch_size)
            # Запуск оптимизации потерь
            c,_ = sess.run([loss,optimizer], feed_dict={input_tensor: batch_x,output_tensor:batch_y})
            # Вычисление средних потерь
            avg_cost += c / total_batch
        # Информация о каждой эпохе
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "loss=",
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Тестовая модель
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(output_tensor, 1))
    # Вычисление точности предсказания
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    total_test_data = len(newsgroups_test.target)
    batch_x_test, batch_y_test = get_batch(newsgroups_test,0,total_test_data)
    print("Accuracy:", accuracy.eval({input_tensor: batch_x_test, output_tensor: batch_y_test}))
