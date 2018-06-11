import tensorflow as tf
import numpy as np
import time
import datetime
import os
import network
from context_split import context_split
from context_split import context_mask
from sklearn.metrics import average_precision_score

FLAGS = tf.app.flags.FLAGS


def slice_cascade_data(data, start, end):
    plate = []
    for piece in data:
        plate.append(piece[start: end])
    return plate


def main(_):
    pathname = './model/PCNN_model-'
    word_embedding = np.load('./data/vec.npy')
    test_settings = network.Settings()
    test_settings.vocab_size = 114044
    test_settings.num_classes = test_num_classes = 53
    test_settings.entity_count = test_entity_count = 262 * 9

    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            with tf.variable_scope('model'):
                mtest = network.PCNNMasked(is_training=False,
                                           word_embeddings=word_embedding,
                                           settings=test_settings)
            saver = tf.train.Saver()

            def test_step(word_batch, pos1_batch, pos2_batch, y_batch,
                          mask_left, mask_mid, mask_right, entity_count):
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

                feed_dict[mtest.total_shape] = np.array(total_shape)
                feed_dict[mtest.input_word] = np.array(total_word)
                feed_dict[mtest.input_pos1] = np.array(total_pos1)
                feed_dict[mtest.input_pos2] = np.array(total_pos2)
                feed_dict[mtest.input_y] = y_batch
                feed_dict[mtest.mask_left] = np.array(total_mask_left)
                feed_dict[mtest.mask_mid] = np.array(total_mask_mid)
                feed_dict[mtest.mask_right] = np.array(total_mask_right)

                loss, accuracy, prob = sess.run(
                    [mtest.loss, mtest.accuracy, mtest.prob], feed_dict)
                return prob, accuracy

            def generate_prob(test_y, test_word, test_pos1, test_pos2,
                              test_mask_l, test_mask_m, test_mask_r, mtest_settings):
                all_prob = []
                acc = []
                entity_count = test_settings.entity_count
                for i in range(len(test_word) // entity_count):
                    prob, accuracy = test_step(
                        test_word[i * entity_count: (i + 1) * entity_count],
                        test_pos1[i * entity_count: (i + 1) * entity_count],
                        test_pos2[i * entity_count: (i + 1) * entity_count],
                        test_y[i * test_entity_count: (i + 1) * entity_count],
                        test_mask_l[i * entity_count: (i + 1) * entity_count],
                        test_mask_m[i * entity_count: (i + 1) * entity_count],
                        test_mask_r[i * entity_count: (i + 1) * entity_count],
                        entity_count
                    )
                    acc.append(np.mean(np.reshape(np.array(accuracy),
                                                  entity_count)))
                    prob = np.reshape(np.array(prob),
                                      (entity_count, test_num_classes))
                    for single_prob in prob:
                        all_prob.append(single_prob[1:])

                all_prob = np.reshape(np.array(all_prob), (-1))
                return all_prob

            def print_pn(all_ans, all_prob):
                order = np.argsort(-all_prob)

                print('P@100:')
                top100 = order[:100]
                correct_num_100 = 0.0
                for i in top100:
                    if all_ans[i] == 1:
                        correct_num_100 += 1.0
                print(correct_num_100 / 100)

                print('P@200:')
                top200 = order[:200]
                correct_num_200 = 0.0
                for i in top200:
                    if all_ans[i] == 1:
                        correct_num_200 += 1.0
                print(correct_num_200 / 200)

                print('P@300:')
                top300 = order[:300]
                correct_num_300 = 0.0
                for i in top300:
                    if all_ans[i] == 1:
                        correct_num_300 += 1.0
                print(correct_num_300 / 300)

            def eval_pn(test_y, test_word, test_pos1, test_pos2,
                        test_mask_l, test_mask_m, test_mask_r, test_settings):
                all_prob = generate_prob(
                    test_y, test_word, test_pos1, test_pos2,
                    test_mask_l, test_mask_m, test_mask_r, test_settings)
                eval_y = []
                for i in test_y:
                    eval_y.append(i[1:])
                all_ans = np.reshape(eval_y, -1)
                print_pn(all_ans, all_prob)

            test_list = [17000]

            for model_iter in test_list:
                saver.restore(sess, pathname + str(model_iter))
                print('Restore Complete')
                print('Evaluating P@N for iter' + str(model_iter))

                print('Evaluating P@N for one:')
                test_y = np.load('./data/pone_test_y.npy')
                test_word = np.load('./data/pone_test_word.npy')
                test_pos1 = np.load('./data/pone_test_pos1.npy')
                test_pos2 = np.load('./data/pone_test_pos2.npy')
                mask_left, mask_mid, mask_right, c_word, c_pos1, c_pos2, c_y \
                    = context_mask(test_word, test_pos1, test_pos2, test_y)
                eval_pn(
                    c_y, c_word, c_pos1, c_pos2, mask_left,
                    mask_mid, mask_right, test_settings)

                print('Evaluating P@N for two:')
                test_y = np.load('./data/ptwo_test_y.npy')
                test_word = np.load('./data/ptwo_test_word.npy')
                test_pos1 = np.load('./data/ptwo_test_pos1.npy')
                test_pos2 = np.load('./data/ptwo_test_pos2.npy')
                mask_left, mask_mid, mask_right, c_word, c_pos1, c_pos2, c_y \
                    = context_mask(test_word, test_pos1, test_pos2, test_y)
                eval_pn(
                    c_y, c_word, c_pos1, c_pos2, mask_left,
                    mask_mid, mask_right, test_settings)

                print('Evaluating P@N for all:')
                test_y = np.load('./data/pall_test_y.npy')
                test_word = np.load('./data/pall_test_word.npy')
                test_pos1 = np.load('./data/pall_test_pos1.npy')
                test_pos2 = np.load('./data/pall_test_pos2.npy')
                mask_left, mask_mid, mask_right, c_word, c_pos1, c_pos2, c_y \
                    = context_mask(test_word, test_pos1, test_pos2, test_y)
                eval_pn(
                    c_y, c_word, c_pos1, c_pos2, mask_left,
                    mask_mid, mask_right, test_settings)

                time_str = datetime.datetime.now().isoformat()
                print(time_str)

                print('Evaluating all test data and save data for PR curve')
                test_y = np.load('./data/testall_y.npy')
                test_word = np.load('./data/testall_word.npy')
                test_pos1 = np.load('./data/testall_pos1.npy')
                test_pos2 = np.load('./data/testall_pos2.npy')
                mask_left, mask_mid, mask_right, c_word, c_pos1, c_pos2, c_y \
                    = context_mask(test_word, test_pos1, test_pos2, test_y)
                all_prob_ = generate_prob(
                    c_y, c_word, c_pos1, c_pos2,
                    mask_left, mask_mid, mask_right, test_settings)
                # all_ans_ = np.load('./data/allans.npy')
                eval_y = []
                for i in c_y:
                    eval_y.append(i[1:])
                all_ans_ = np.reshape(eval_y, -1)

                print('P@N for all test data:')
                print_pn(all_ans_, all_prob_)

                print('saving all test result...')
                current_step = model_iter
                np.save('./out/all_prob_iter_' + str(current_step) + '.npy',
                        all_prob_)

                # print(np.shape(all_prob_), np.shape(all_ans_))
                # length of all_prob_ is shorter than all_ans_
                # because of batching

                all_ans_trimmed = all_ans_[:all_prob_.size]
                avg_precision = average_precision_score(all_ans_trimmed, all_prob_)
                print('PR curve area:', str(avg_precision))

                time_str = datetime.datetime.now().isoformat()
                print(time_str)


if __name__ == '__main__':
    tf.app.run()
