import tensorflow as tf
import numpy as np


class Settings(object):
    def __init__(self):
        self.vocab_size = 114042
        self.word_emb_size = 50
        self.num_classes = 53
        self.input_size = 70
        self.window_size = 6
        self.num_filters = 3
        self.hidden_size = 100
        self.dense_size = 200
        self.num_maps = 4
        self.epochs = 3
        self.keep_prob = 0.5
        self.num_layers = 1
        self.pos_size = 5
        self.pos_num = 123
        # the number of entity pairs of each batch during training or testing
        self.entity_count = 50
        self.num_epochs = 3


class PCNNMasked:
    def __init__(self, is_training, word_embeddings, settings):
        def emb_gen(_word, _pos1, _pos2, w_emb, pos1_emb, pos2_emb):
            """ Generate input vector from inputs and embeddings """
            return tf.concat([
                tf.nn.embedding_lookup(w_emb, _word),
                tf.nn.embedding_lookup(pos1_emb, _pos1),
                tf.nn.embedding_lookup(pos2_emb, _pos2)
            ], 2)

        def conv_layer(_input, _window_size, _output_dim, pool='MAX'):
            input_dim = _input.get_shape().as_list()[-1]
            conv_weight = tf.get_variable(
                'conv_w', [_window_size, input_dim, _output_dim]
            )
            conv_out = tf.nn.conv1d(
                _input, conv_weight, stride=1, padding='SAME'
            )
            if pool == 'MAX':
                return tf.reduce_max(conv_out, axis=1)
            if pool == 'AVG':
                return tf.reduce_mean(conv_out, axis=1)
            if pool == 'NONE':
                return conv_out

        def mask_pool(_input, _mask0, _mask1, _mask2):
            _input0 = tf.multiply(_input, _mask0)
            _input1 = tf.multiply(_input, _mask1)
            _input2 = tf.multiply(_input, _mask2)
            pool0 = tf.reduce_max(_input0, axis=1)
            pool1 = tf.reduce_max(_input1, axis=1)
            pool2 = tf.reduce_max(_input2, axis=1)
            return pool0, pool1, pool2

        self.vocab_size = vocab_size = settings.vocab_size
        self.num_classes = num_classes = settings.num_classes
        self.entity_count = entity_count = settings.entity_count
        self.hidden_size = hidden_size = settings.hidden_size
        self.input_size = input_size = settings.input_size
        self.window_size = window_size = settings.window_size
        self.num_filters = num_filters = settings.num_filters
        self.dense_size = dense_size = settings.dense_size
        self.keep_prob = keep_prob = settings.keep_prob
        self.emb_size = emb_size = \
            settings.word_emb_size + settings.pos_size * 2

        self.input_word = tf.placeholder(dtype=tf.int32,
                                         shape=[None, input_size],
                                         name='input_word')
        self.input_pos1 = tf.placeholder(dtype=tf.int32,
                                         shape=[None, input_size],
                                         name='input_pos1')
        self.input_pos2 = tf.placeholder(dtype=tf.int32,
                                         shape=[None, input_size],
                                         name='input_pos2')
        self.input_y = tf.placeholder(dtype=tf.float32,
                                      shape=[None, num_classes],
                                      name='input_y')
        self.total_shape = tf.placeholder(dtype=tf.int32,
                                          shape=[entity_count + 1],
                                          name='total_shape')
        self.mask_left = tf.placeholder(dtype=tf.float32,
                                        shape=[None, input_size],
                                        name='mask_left')
        self.mask_mid = tf.placeholder(dtype=tf.float32,
                                       shape=[None, input_size],
                                       name='mask_mid')
        self.mask_right = tf.placeholder(dtype=tf.float32,
                                         shape=[None, input_size],
                                         name='mask_right')
        total_num = self.total_shape[-1]
        word_embedding = tf.get_variable(initializer=word_embeddings,
                                         name='word_embedding')
        pos1_embedding = tf.get_variable('pos1_embedding',
                                         [settings.pos_num, settings.pos_size])
        pos2_embedding = tf.get_variable('pos2_embedding',
                                         [settings.pos_num, settings.pos_size])
        self.relation_embedding = tf.get_variable(
            'relation_embedding', [self.num_classes, dense_size])
        inputs = emb_gen(
            self.input_word, self.input_pos1, self.input_pos2,
            word_embedding, pos1_embedding, pos2_embedding
        )
        mask_left = tf.expand_dims(self.mask_left, -1)
        mask_mid = tf.expand_dims(self.mask_mid, -1)
        mask_right = tf.expand_dims(self.mask_right, -1)

        # Piecewise Pooling CNN
        conv_left = []
        conv_mid = []
        conv_right = []
        for i in range(num_filters):
            with tf.variable_scope('conv_map_{}'.format(i)):
                conv_out = conv_layer(
                    inputs, window_size, hidden_size, pool='NONE')
                pool_left, pool_mid, pool_right = mask_pool(
                    conv_out, mask_left, mask_mid, mask_right)
                conv_left.append(pool_left)
                conv_mid.append(pool_mid)
                conv_right.append(pool_right)

        context_left = tf.concat(conv_left, axis=1)
        context_mid = tf.concat(conv_mid, axis=1)
        context_right = tf.concat(conv_right, axis=1)
        context_output = tf.nn.relu(
            tf.concat([context_left, context_mid, context_right], 1)
        )
        if is_training:
            context_output = tf.nn.dropout(context_output, keep_prob)

        dense_input_size = settings.num_filters * hidden_size * 3
        dense_weight = tf.get_variable('dense_weight',
                                       [dense_input_size, dense_size])
        dense_bias = tf.get_variable('dense_bias', [dense_size])
        dense_output = tf.nn.xw_plus_b(context_output, dense_weight, dense_bias)
        dense_output = tf.nn.relu(dense_output)

        # Multiple Instance Learning
        bag_input = []
        bag_alpha = []
        bag_output = []
        bag_logits = []
        # bag_prob = []
        self.predictions = []
        self.prob = []
        self.loss = []
        self.accuracy = []
        self.total_loss = 0.0
        attention_a = tf.get_variable('attention_a', [dense_size])
        attention_w = tf.get_variable('attention_w', [dense_size, 1])
        bag_bias = tf.get_variable('bag_bias', [self.num_classes])
        for i in range(entity_count):
            bag_input.append(
                dense_output[self.total_shape[i]:self.total_shape[i+1]])
            # bag_size = self.total_shape[i+1] - self.total_shape[i]
            # bag_alpha[i]: (1, bag_size)
            bag_alpha.append(tf.nn.softmax(
                tf.transpose(
                    tf.matmul(
                        tf.multiply(bag_input[i], attention_a), attention_w
                    ),
                )
            ))
            # bag_output[i]: (dense_size, 1)
            bag_output.append(
                tf.transpose(tf.matmul(bag_alpha[i], bag_input[i])))
            bag_logits.append(
                tf.add(tf.reshape(
                    tf.matmul(self.relation_embedding, bag_output[i]),
                    [self.num_classes]
                ), bag_bias)
            )
            self.prob.append(tf.nn.softmax(bag_logits[i]))
            # bag_prob.append(tf.nn.softmax(bag_logits[i]))

            with tf.name_scope('output'):
                self.predictions.append(
                    tf.argmax(self.prob[i], 0, name='predictions'))

            with tf.name_scope('loss'):
                self.loss.append(tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        logits=bag_logits[i],
                        labels=self.input_y[i]
                    )
                ))
                if i == 0:
                    self.total_loss = self.loss[i]
                else:
                    self.total_loss += self.loss[i]
            with tf.name_scope('accuracy'):
                self.accuracy.append(tf.reduce_mean(
                    tf.cast(
                        tf.equal(self.predictions[i],
                                 tf.argmax(self.input_y[i],0)),
                        'float'
                    ), name='accuracy'
                ))

        # Regularization
        self.l2_loss = tf.contrib.layers.apply_regularization(
            regularizer=tf.contrib.layers.l2_regularizer(0.0001),
            weights_list=tf.trainable_variables()
        )
        self.final_loss = self.total_loss + self.l2_loss

        tf.summary.scalar('loss', self.total_loss)
        tf.summary.scalar('l2_loss', self.l2_loss)
        tf.summary.scalar('final_loss', self.final_loss)

class PCNN:
    def __init__(self, is_training, word_embeddings, settings):
        def emb_gen(_word, _pos1, _pos2, w_emb, pos1_emb, pos2_emb):
            """ Generate input vector from inputs and embeddings """
            return tf.concat([
                tf.nn.embedding_lookup(w_emb, _word),
                tf.nn.embedding_lookup(pos1_emb, _pos1),
                tf.nn.embedding_lookup(pos2_emb, _pos2)
            ], 2)

        def conv_layer(_conv_input, _filter):
            """
            Apply convolution and max-polling to _conv_input and _filter
            :param _conv_input: Tensor w/ shape [batch_len, length, emb_size]
            :param _filter: Tensor w/ shape [window_size, emb_size, hidden_size]
            :return: Tensor w/ shape [batch_len, hidden_size]
            """
            _conv_output = tf.nn.conv1d(
                _conv_input, filters=_filter, stride=1, padding='SAME')
            return tf.reduce_max(_conv_output, axis=1)

        self.window_size = window_size = settings.window_size
        self.vocab_size = vocab_size = settings.vocab_size
        self.num_classes = num_classes = settings.num_classes
        self.entity_count = entity_count = settings.entity_count
        self.hidden_size = hidden_size = settings.hidden_size
        self.input_size = input_size = settings.input_size
        self.dense_size = dense_size = settings.dense_size
        self.keep_prob = keep_prob = settings.keep_prob
        self.emb_size = emb_size = \
            settings.word_emb_size + settings.pos_size * 2

        # size=[batch_size, left_length]
        self.input_word_left = tf.placeholder(dtype=tf.int32,
                                              shape=[None, None],
                                              name='input_word_left')
        self.input_word_mid = tf.placeholder(dtype=tf.int32,
                                             shape=[None, None],
                                             name='input_word_mid')
        self.input_word_right = tf.placeholder(dtype=tf.int32,
                                               shape=[None, None],
                                               name='input_word_right')
        self.input_pos1_left = tf.placeholder(dtype=tf.int32,
                                              shape=[None, None],
                                              name='input_pos1_left')
        self.input_pos1_mid = tf.placeholder(dtype=tf.int32,
                                             shape=[None, None],
                                             name='input_pos1_mid')
        self.input_pos1_right = tf.placeholder(dtype=tf.int32,
                                               shape=[None, None],
                                               name='input_pos1_right')
        self.input_pos2_left = tf.placeholder(dtype=tf.int32,
                                              shape=[None, None],
                                              name='input_pos2_left')
        self.input_pos2_mid = tf.placeholder(dtype=tf.int32,
                                             shape=[None, None],
                                             name='input_pos2_mid')
        self.input_pos2_right = tf.placeholder(dtype=tf.int32,
                                               shape=[None, None],
                                               name='input_pos2_right')
        self.input_y = tf.placeholder(dtype=tf.float32,
                                      shape=[None, num_classes],
                                      name='input_y')
        self.total_shape = tf.placeholder(dtype=tf.int32,
                                          shape=[entity_count + 1],
                                          name='total_shape')
        total_num = self.total_shape[-1]

        word_embedding = tf.get_variable(initializer=word_embeddings,
                                         name='word_embedding')
        pos1_embedding = tf.get_variable('pos1_embedding',
                                         [settings.pos_num, settings.pos_size])
        pos2_embedding = tf.get_variable('pos2_embedding',
                                         [settings.pos_num, settings.pos_size])
        self.relation_embedding = tf.get_variable('relation_embedding',
                                                  [self.num_classes, dense_size])
        inputs_left = emb_gen(
            self.input_word_left, self.input_pos1_left, self.input_pos2_left,
            word_embedding, pos1_embedding, pos2_embedding
        )
        inputs_mid = emb_gen(
            self.input_word_mid, self.input_pos1_mid, self.input_pos2_mid,
            word_embedding, pos1_embedding, pos2_embedding
        )
        inputs_right = emb_gen(
            self.input_word_right, self.input_pos1_right, self.input_pos2_right,
            word_embedding, pos1_embedding, pos2_embedding
        )

        conv_left = []
        conv_mid = []
        conv_right = []
        with tf.variable_scope('conv_maps'):
            for i in range(settings.num_maps):
                with tf.variable_scope('conv_map' + str(i)):
                    conv_weight = tf.get_variable(
                        'conv_w', [window_size, emb_size, hidden_size])
                    conv_left.append(conv_layer(inputs_left, conv_weight))
                    conv_mid.append(conv_layer(inputs_mid, conv_weight))
                    conv_right.append(conv_layer(inputs_right, conv_weight))

        context_left = tf.concat(conv_left, axis=1)
        context_mid = tf.concat(conv_mid, axis=1)
        context_right = tf.concat(conv_right, axis=1)
        context_output = tf.tanh(
            tf.concat([context_left, context_mid, context_right], 1)
        )
        if is_training:
            context_output = tf.nn.dropout(context_output, keep_prob)

        dense_input_size = settings.num_maps * hidden_size * 3
        dense_weight = tf.get_variable('dense_weight',
                                       [dense_input_size, dense_size])
        dense_bias = tf.get_variable('dense_bias', [dense_size])
        dense_output = tf.nn.xw_plus_b(context_output, dense_weight, dense_bias)
        dense_output = tf.tanh(dense_output)

        # Multiple Instance Learning
        # Sentence-level attention
        bag_input = []
        bag_alpha = []
        bag_output = []
        bag_logits = []
        # bag_prob = []
        self.predictions = []
        self.prob = []
        self.loss = []
        self.accuracy = []
        self.total_loss = 0.0
        attention_a = tf.get_variable('attention_a', [dense_size])
        attention_w = tf.get_variable('attention_w', [dense_size, 1])
        bag_bias = tf.get_variable('bag_bias', [self.num_classes])
        for i in range(entity_count):
            bag_input.append(
                dense_output[self.total_shape[i]:self.total_shape[i+1]])
            # bag_size = self.total_shape[i+1] - self.total_shape[i]
            # bag_alpha[i]: (1, bag_size)
            bag_alpha.append(tf.nn.softmax(
                tf.transpose(
                    tf.matmul(
                        tf.multiply(bag_input[i], attention_a), attention_w
                    ),
                )
            ))
            # bag_output[i]: (dense_size, 1)
            bag_output.append(
                tf.transpose(tf.matmul(bag_alpha[i], bag_input[i])))
            bag_logits.append(
                tf.add(tf.reshape(
                    tf.matmul(self.relation_embedding, bag_output[i]),
                    [self.num_classes]
                ), bag_bias)
            )
            self.prob.append(tf.nn.softmax(bag_logits[i]))
            # bag_prob.append(tf.nn.softmax(bag_logits[i]))

            with tf.name_scope('output'):
                self.predictions.append(
                    tf.argmax(self.prob[i], 0, name='predictions'))

            with tf.name_scope('loss'):
                self.loss.append(tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        logits=bag_logits[i],
                        labels=self.input_y[i]
                    )
                ))
                if i == 0:
                    self.total_loss = self.loss[i]
                else:
                    self.total_loss += self.loss[i]
            with tf.name_scope('accuracy'):
                self.accuracy.append(tf.reduce_mean(
                    tf.cast(
                        tf.equal(self.predictions[i],
                                 tf.argmax(self.input_y[i],0)),
                        'float'
                    ), name='accuracy'
                ))

        # Regularization
        self.l2_loss = tf.contrib.layers.apply_regularization(
            regularizer=tf.contrib.layers.l2_regularizer(0.0001),
            weights_list=tf.trainable_variables()
        )
        self.final_loss = self.total_loss + self.l2_loss

        tf.summary.scalar('loss', self.total_loss)
        tf.summary.scalar('l2_loss', self.l2_loss)
        tf.summary.scalar('final_loss', self.final_loss)
