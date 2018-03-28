import tensorflow as tf
import numpy as np
import time
import datetime
import os
import network
from context_split import context_split

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('summary_dir', '.', 'path to store summary')


def main(_):
    save_path = './model/'

    print('reading word embedding')
    word_embedding = np.load('./data/vec.npy')

    print('reading corpus')
    train_y = np.load('./data/small_y.npy')
    train_word = np.load('./data/small_word.npy')
    train_pos1 = np.load('./data/small_pos1.npy')
    train_pos2 = np.load('./data/small_pos2.npy')

    context_word, context_pos1, context_pos2 = context_split(
        train_word, train_pos1, train_pos2
    )

    settings = network.Settings()
    settings.vocab_size = len(word_embedding)
    settings.num_classes = len(train_y[0])

    print(settings.num_classes)

    entity_count = settings.entity_count

    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            initializer = tf.contrib.layers.xavier_initializer()
            with tf.variable_scope('model', reuse=None,
                                   initializer=initializer):
                m = network.PCNN(is_training=True,
                                 word_embeddings=word_embedding,
                                 settings=settings)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer(0.001)
            train_op = optimizer.minimize(m.final_loss, global_step=global_step)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=None)

            merged_summary = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(
                FLAGS.summary_dir + '/train_loss', sess.graph
            )

            def train_step(word_batch, pos1_batch, pos2_batch, y_batch,
                           entity_count):
                feed_dict = {}
                total_shape = []
                total_num = 0
                total_word = [[], [], []]
                total_pos1 = [[], [], []]
                total_pos2 = [[], [], []]
                for i in range(len(word_batch[0])):
                    total_shape.append(total_num)
                    total_num += len(word_batch[0][i])
                    for loc in range(3):
                        for word in word_batch[loc][i]:
                            total_word[loc].append(word)
                        for pos1 in pos1_batch[loc][i]:
                            total_pos1[loc].append(pos1)
                        for pos2 in pos2_batch[loc][i]:
                            total_pos2[loc].append(pos2)
                total_shape.append(total_num)
                total_shape = np.array(total_shape)

                feed_dict[m.total_shape] = np.array(total_shape)

                feed_dict[m.input_word_left] = np.array(total_word[0])
                feed_dict[m.input_word_mid] = np.array(total_word[1])
                feed_dict[m.input_word_right] = np.array(total_word[2])

                feed_dict[m.input_pos1_left] = np.array(total_pos1[0])
                feed_dict[m.input_pos1_mid] = np.array(total_pos1[1])
                feed_dict[m.input_pos1_right] = np.array(total_pos1[2])

                feed_dict[m.input_pos2_left] = np.array(total_pos2[0])
                feed_dict[m.input_pos2_mid] = np.array(total_pos2[1])
                feed_dict[m.input_pos2_right] = np.array(total_pos2[2])

                feed_dict[m.input_y] = y_batch

                _, step, loss, accuracy, summary, l2_loss, final_loss = \
                    sess.run(
                        [train_op, global_step, m.total_loss, m.accuracy,
                         merged_summary, m.l2_loss, m.final_loss],
                        feed_dict)
                time_str = datetime.datetime.now().isoformat()
                accuracy = np.reshape(np.array(accuracy), entity_count)
                acc = np.mean(accuracy)
                summary_writer.add_summary(summary, step)

                if step % 50 == 0:
                    tmp_str = '{}: step{}, softmax_loss {:g}, acc {:g}'.format(
                        time_str, step, loss, acc
                    )
                    print(tmp_str)

            for one_epoch in range(settings.num_epochs):
                tmp_order = np.arange(len(context_word[0]))
                np.random.shuffle(tmp_order)
                for i in range(len(tmp_order) // settings.entity_count):
                    tmp_word = [[], [], []]
                    tmp_pos1 = [[], [], []]
                    tmp_pos2 = [[], [], []]
                    tmp_y = []
                    tmp_input = tmp_order[
                        i*settings.entity_count:(i + 1)*settings.entity_count
                    ]
                    for k in tmp_input:
                        for loc in range(3):
                            tmp_word[loc].append(context_word[loc][k])
                            tmp_pos1[loc].append(context_pos1[loc][k])
                            tmp_pos2[loc].append(context_pos2[loc][k])
                        tmp_y.append(train_y[k])
                    num = 0
                    for single_word in tmp_word[0]:
                        num += len(single_word)

                    if num > 1500:
                        print('out of range')
                        continue

                    train_step(tmp_word, tmp_pos1, tmp_pos2, tmp_y,
                               settings.entity_count)

                    current_step = tf.train.global_step(sess, global_step)
                    if current_step  > 9000 and current_step % 500 == 0:
                        print('saving model')
                        path = saver.save(sess, save_path + 'PCNN_model',
                                          global_step=current_step)
                        tmpstr = 'saved model to ' + path
                        print(tmpstr)


if __name__ == '__main__':
    tf.app.run()
