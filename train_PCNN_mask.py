import tensorflow as tf
import numpy as np
import time
import datetime
import os
import network
from context_split import context_split
from context_split import context_mask

FLAGS = tf.app.flags.FLAGS

def main(_):
    time_start = time.time()

    save_path = './model/'

    print('reading word embedding')
    word_embedding = np.load('./data/vec.npy')

    print('reading corpus')
    train_y = np.load('./data/small_y.npy')
    train_word = np.load('./data/small_word.npy')
    train_pos1 = np.load('./data/small_pos1.npy')
    train_pos2 = np.load('./data/small_pos2.npy')

    mask_left, mask_mid, mask_right, train_word, train_pos1, train_pos2, train_y \
        = context_mask(train_word, train_pos1, train_pos2, train_y)

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
                m = network.PCNNMasked(is_training=True,
                                       word_embeddings=word_embedding,
                                       settings=settings)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer(0.001)
            train_op = optimizer.minimize(m.final_loss, global_step=global_step)
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver(max_to_keep=None)
            time_str = datetime.datetime.now().isoformat()
            tim_str = time_str[:time_str.rfind('.')]
            merged_summary = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(
                FLAGS.summary_dir + '/train_loss', sess.graph
            )

            def train_step(word_batch, pos1_batch, pos2_batch, y_batch,
                           mask_left, mask_mid, mask_right ,entity_count):
                feed_dict = {}
                total_shape = []
                total_num = 0
                total_word = []
                total_pos1 = []
                total_pos2 = []
                total_mask_left = []
                total_mask_mid = []
                total_mask_right = []
                for i, word_seq in enumerate(word_batch):
                    total_shape.append(total_num)
                    total_num += len(word_seq)
                    for word in word_seq:
                        total_word.append(word)
                    for pos1 in pos1_batch[i]:
                        total_pos1.append(pos1)
                    for pos2 in pos2_batch[i]:
                        total_pos2.append(pos2)
                    for mask in mask_left[i]:
                        total_mask_left.append(mask)
                    for mask in mask_mid[i]:
                        total_mask_mid.append(mask)
                    for mask in mask_right[i]:
                        total_mask_right.append(mask)

                total_shape.append(total_num)

                feed_dict[m.total_shape] = np.array(total_shape)
                feed_dict[m.input_word] = np.array(total_word)
                feed_dict[m.input_pos1] = np.array(total_pos1)
                feed_dict[m.input_pos2] = np.array(total_pos2)
                feed_dict[m.input_y] = y_batch
                feed_dict[m.mask_left] = np.array(total_mask_left)
                feed_dict[m.mask_mid] = np.array(total_mask_mid)
                feed_dict[m.mask_right] = np.array(total_mask_right)

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
                tmp_order = np.arange(len(train_word))
                np.random.shuffle(tmp_order)
                for i in range(len(tmp_order) // settings.entity_count):
                    tmp_word = []
                    tmp_pos1 = []
                    tmp_pos2 = []
                    tmp_y = []
                    tmp_mask_l = []
                    tmp_mask_m = []
                    tmp_mask_r = []
                    tmp_input = tmp_order[
                        i*settings.entity_count:(i + 1)*settings.entity_count
                    ]
                    for k in tmp_input:
                        tmp_word.append(train_word[k])
                        tmp_pos1.append(train_pos1[k])
                        tmp_pos2.append(train_pos2[k])
                        tmp_y.append(train_y[k])
                        tmp_mask_l.append(mask_left[k])
                        tmp_mask_m.append(mask_mid[k])
                        tmp_mask_r.append(mask_right[k])
                    num = 0
                    for single_word in tmp_word[0]:
                        num += len(single_word)

                    if num > 1500:
                        print('out of range')
                        continue

                    train_step(
                        tmp_word, tmp_pos1, tmp_pos2, tmp_y,
                        tmp_mask_l,
                        tmp_mask_m,
                        tmp_mask_r,
                        settings.entity_count
                    )

                    current_step = tf.train.global_step(sess, global_step)
                    if current_step > 15000 and current_step % 500 == 0:
                        print('saving model')
                        path = saver.save(sess, save_path + 'PCNN_model',
                                          global_step=current_step)
                        tmpstr = 'saved model to ' + path
                        print(tmpstr)
                current_step = tf.train.global_step(sess, global_step)
                print('saving model')
                path = saver.save(
                    sess,
                    save_path + 'Hybrid_model_epoch{}'.format(one_epoch+1),
                    global_step=current_step)
    time_finish = time.time()
    time_elapsed = time_finish - time_start
    print('Time Used:', str(datetime.timedelta(seconds=time_elapsed)))


if __name__ == '__main__':
    tf.app.flags.DEFINE_string('summary_dir', '.', 'path to store summary')
    time_str = datetime.datetime.now().isoformat()
    print(time_str)
    tf.app.run()
    time_str = datetime.datetime.now().isoformat()
    print(time_str)
