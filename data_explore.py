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


def main():
    explore_emb()


if __name__ == '__main__':
    main()

