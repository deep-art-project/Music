from functools import reduce
import numpy as np
import pickle


def poem_to_tensor(filepath):

    '''
    Read poems from file.
    '''
    with open(filepath, 'r') as f:
        lines = f.readlines()
    f.close()
    corpus = []
    for l in lines:
        l = l.strip().split('\t')
        l_ = []
        for sen in l:
            l_.extend(sen.split(' '))
        if len(l_) < 28:
            corpus.append(l_)

    '''
    Get all Chinese characters used in poems.
    '''
    letters = []
    for p in corpus:
        letters.extend(p)
    letters = list(set(letters))
    for i in range(len(letters)):
        if letters[i] == '<R>':
            break
    del letters[i]
    letters.append('<R>')

    '''
    Encode all poems to LongTensor.
    '''
    corpus_ = []
    for p in corpus:
        corpus_.append(list(map(lambda x: letters.index(x) + 1, p)))
    corpus_data = np.array(corpus_)

    '''
    Save preprocessed file.
    '''
    np.save('corpus', corpus_data)
    f = open('chars.pkl', 'wb')
    pickle.dump(letters, f)

def tensor_to_poem(input_x, chars):
    poem = []
    sen_len = 5
    for i, x in enumerate(input_x):
        if i:
            if i % sen_len == 0:
                poem.append('\n')
        poem.append(chars[x - 1])
    poem_ = ''
    poem_ = reduce((lambda x, y: x + y), poem)
    print(poem_)
    return poem_
