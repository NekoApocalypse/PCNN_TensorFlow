import numpy as np


def mask_gen(origin, pos1, pos2):
    mask_left = [0] * len(origin)
    mask_mid = [0] * len(origin)
    mask_right = [0] * len(origin)
    mask_left[:pos1 + 1] = [1] * (pos1 + 1)
    mask_mid[pos1: pos2 + 1] = [1] * (pos2 + 1 - pos1)
    mask_right[pos2:] = [1] * (len(origin) - pos2)
    assert len(origin) == len(mask_left)
    assert len(origin) == len(mask_mid)
    assert len(origin) == len(mask_right)
    return mask_left, mask_mid, mask_left


def sentence_pad(origin, pos1, pos2, pad_token):
    part_left = [pad_token] * len(origin)
    part_mid = [pad_token] * len(origin)
    part_right = [pad_token] * len(origin)

    part_left[:pos1 + 1] = origin[:pos1 + 1]
    part_mid[pos1: pos2 + 1] = origin[pos1: pos2 + 1]
    part_right[pos2:] = origin[pos2:]
    return part_left, part_mid, part_right


def context_mask(train_word, train_pos1, train_pos2, train_y):
    skipped_sentences = 0
    skipped_entity_pair = 0
    mask_left = []
    mask_mid = []
    mask_right = []
    context_x = []
    context_y = []
    context_pos1 = []
    context_pos2 = []
    for i in range(len(train_word)):
        buffer_left = []
        buffer_mid = []
        buffer_right = []
        buffer_x = []
        buffer_pos1 = []
        buffer_pos2 = []
        for j in range(len(train_word[i])):
            if not ((61 in train_pos1[i][j]) and (61 in train_pos2[i][j])):
                # print(i, j)
                skipped_sentences += 1
                continue
            # Warning:
            # training data contains python lists at the last dimension,
            # but testing data contains np.ndarray at the last dimension.
            if isinstance(train_pos1[i][j], np.ndarray):
                pos1_id = train_pos1[i][j].tolist().index(61)
                pos2_id = train_pos2[i][j].tolist().index(61)
            else:
                pos1_id = train_pos1[i][j].index(61)
                pos2_id = train_pos2[i][j].index(61)

            tmp_left, tmp_mid, tmp_right = mask_gen(
                train_word[i][j], pos1_id, pos2_id)
            buffer_left.append(tmp_left)
            buffer_mid.append(tmp_mid)
            buffer_right.append(tmp_right)
            buffer_x.append(train_word[i][j])
            buffer_pos1.append(train_pos1[i][j])
            buffer_pos2.append(train_pos2[i][j])

        if not buffer_left:
            skipped_entity_pair += 1
            continue

        assert len(buffer_left) == len(buffer_x)
        mask_left.append(buffer_left)
        mask_mid.append(buffer_mid)
        mask_right.append(buffer_right)
        context_x.append(buffer_x)
        context_pos1.append(buffer_pos1)
        context_pos2.append(buffer_pos2)
        context_y.append(train_y[i])

    assert len(mask_left) == len(context_x)
    print('Skipped sentences: ', skipped_sentences)
    print('Skipped entity pairs: ', skipped_entity_pair)
    return mask_left, mask_mid, mask_right, context_x, context_pos1, context_pos2, context_y


def context_split(train_word, train_pos1, train_pos2, train_y):
    # [num_position, num_entity, num_sentences, sentence_len]
    # special_tokens: 'BLANK', 'UNK'
    special_token = np.load('./data/special_token.npy')
    blank_token = special_token[0]

    context_word = [[], [], []]
    context_pos1 = [[], [], []]
    context_pos2 = [[], [], []]
    context_y = []
    skipped_sentences = 0
    skipped_entity_pair = 0
    input_length = len(train_word[0][0])
    for i in range(len(train_word)):
        # each buffer contains three list of sentences.
        word_buffer = [[], [], []]
        pos1_buffer = [[], [], []]
        pos2_buffer = [[], [], []]
        for j in range(len(train_word[i])):
            if not ((61 in train_pos1[i][j]) and (61 in train_pos2[i][j])):
                # print(i, j)
                skipped_sentences += 1
                continue

            # Warning:
            # training data contains python lists at the last dimension,
            # but testing data contains np.ndarray at the last dimension.
            # Note:
            # The output of sentence_pad should all be lists.
            if isinstance(train_pos1[i][j], np.ndarray):
                pos1_id = train_pos1[i][j].tolist().index(61)
                pos2_id = train_pos2[i][j].tolist().index(61)
            else:
                pos1_id = train_pos1[i][j].index(61)
                pos2_id = train_pos2[i][j].index(61)
            # each part is padded to input_length
            word_left, word_mid, word_right = \
                sentence_pad(train_word[i][j], pos1_id, pos2_id, blank_token)
            word_buffer[0].append(word_left)
            word_buffer[1].append(word_mid)
            word_buffer[2].append(word_right)

            pos1_left, pos1_mid, pos1_right = \
                sentence_pad(train_pos1[i][j], pos1_id, pos2_id, blank_token)
            pos1_buffer[0].append(pos1_left)
            pos1_buffer[1].append(pos1_mid)
            pos1_buffer[2].append(pos1_right)

            pos2_left, pos2_mid, pos2_right = \
                sentence_pad(train_pos2[i][j], pos1_id, pos2_id, blank_token)
            pos2_buffer[0].append(pos2_left)
            pos2_buffer[1].append(pos2_mid)
            pos2_buffer[2].append(pos2_right)

        if not word_buffer[0]:
            skipped_entity_pair += 1
            # print(i)
            continue

        for t in range(3):
            context_word[t].append(word_buffer[t])
            context_pos1[t].append(pos1_buffer[t])
            context_pos2[t].append(pos2_buffer[t])
        context_y.append(train_y[i])
    print('Skipped sentences: ', skipped_sentences)
    print('Skipped entity pairs: ', skipped_entity_pair)
    return context_word, context_pos1, context_pos2, context_y
