import numpy as np


def sentence_pad(origin, pos1, pos2, pad_token):
    part_left = [pad_token] * len(origin)
    part_mid = [pad_token] * len(origin)
    part_right = [pad_token] * len(origin)

    part_left[:pos1 + 1] = origin[:pos1 + 1]
    part_mid[pos1: pos2 + 1] = origin[pos1: pos2 + 1]
    part_right[pos2:] = origin[pos2:]
    return part_left, part_mid, part_right


def context_split(train_word, train_pos1, train_pos2):
    # [num_position, num_entity, num_sentences, sentence_len]
    # special_tokens: 'BLANK', 'UNK'
    special_token = np.load('./data/special_token.npy')
    blank_token = special_token[0]

    context_word = [[], [], []]
    context_pos1 = [[], [], []]
    context_pos2 = [[], [], []]
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
    print('Skipped sentences: ', skipped_sentences)
    print('Skipped entity pairs: ', skipped_entity_pair)
    return context_word, context_pos1, context_pos2
