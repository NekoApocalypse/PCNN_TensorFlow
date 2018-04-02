import numpy as np
from context_split import context_split


def explore_input_data():
    print('reading training data')
    train_y = np.load('./data/small_y.npy')
    train_word = np.load('./data/small_word.npy')
    train_pos1 = np.load('./data/small_pos1.npy')
    train_pos2 = np.load('./data/small_pos2.npy')

    context_word, context_pos1, context_pos2 = context_split(
        train_word, train_pos1, train_pos2
    )
    print(train_word[0])
    print(context_word[0][0])


def explore_emb():
    word_embedding = np.load('./data/vec.npy')
    print(np.shape(word_embedding))


def test_sanity_check():
    test_y = np.load('./data/testall_y.npy')
    test_word = np.load('./data/testall_word.npy')
    test_pos1 = np.load('./data/testall_pos1.npy')
    test_pos2 = np.load('./data/testall_pos2.npy')

    print(len(test_y), len(test_word))

    c_word, c_pos1, c_pos2, c_y = context_split(
        test_word, test_pos1, test_pos2, test_y
    )

    print(len(c_y), len(c_word[0]))

    eval_y = []
    for i in c_y:
        eval_y.append(i[1:])
    all_ans = np.reshape(eval_y, -1)
    print(np.shape(all_ans))
    print(all_ans.size)


def main():
    # explore_emb()
    test_sanity_check()


if __name__ == '__main__':
    main()

