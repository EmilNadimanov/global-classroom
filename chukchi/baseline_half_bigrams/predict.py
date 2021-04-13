import os
import sys
import pickle

# This is the filename of the model file
sys.path.append(os.path.abspath('../..'))
from chukchi.tree.CharTree import CharTree

TREE = CharTree.build_tree(
    open("../data/texts.txt").read(),
    step=2)


def predict_by_chars(word):
    result = list()
    success = False
    for j in range(len(word)):
        if TREE.predict(word[0: j]) == word:
            rest = word[j + 1:]
            result.extend([word[j], rest])
            success = True
            break
        else:
            result.append(word[j])
    return result, success


#model_file = sys.argv[1]
model_file = './model.dat'

mf = open(model_file, 'rb')

# Load the unigram and bigram probabilities from the model file
(unigrams, half_bigrams, bigrams) = pickle.load(mf)
n = 2

# Initialise the probability of the start of the sentence.
unigrams['#'] = 0.0

# Hits is number of times we get a prediction right
hits = 0
# The number of tokens
n_tokens = 0
# For each of the lines in the input
with open(f'../data/test/test.tsv', encoding='utf-8') as f:
    lines = f.readlines()

# df_half_bigrams = pd.DataFrame.from_dict(half_bigrams)
# print(df_half_bigrams.head())

for line in open("../data/test/test.tsv").readlines():
    # for line in lines:
    row = line.strip().split('\t')
    # Our tokens are in column one, split by space
    tokens = row[0].split(' ')
    # The test tokens are the beginning of sentence symbol + the list of tokens
    tst_tokens = ['#'] + tokens
    # Increment the number of tokens by the length of the list containing the
    # tokens
    n_tokens += len(tokens)

    # This is our output
    output = []
    # For each of the tokens in the "tst_tokens" list (e.g. the list + the
    # beginning of sentence symbol)
    for i in range(len(tst_tokens) - 1):
        first = tst_tokens[i]
        second = tst_tokens[i + 1]
        if first in half_bigrams['0']:
            pred = max(
                half_bigrams['0'][first],
                key=half_bigrams['0'][first].get)
            if pred == second:
                output.append(pred)
                hits += 1
            else:
                flag = False
                for n in range(len(second) - 1):
                    if first in bigrams[str(n + 1)]:
                        second_begin = second[:n + 1]
                        if second_begin in bigrams[str(n + 1)][first]:
                            # print(bigrams[str(n + 1)][first][second_begin])
                            pred = max(bigrams[str(
                                n + 1)][first][second_begin], key=bigrams[str(n + 1)][first][second_begin].get)
                            if second_begin + pred == second:
                                output += [c for c in second_begin]
                                output.append(pred)
                                hits += 1
                                flag = True
                                # print('HERE')
                                break
                if flag is False:
                    output += [c for c in second]
        else:
            pred = max(unigrams, key=unigrams.get)
            if pred == second:
                output.append(pred)
                hits += 1
            else:
                (prediction, got_lucky) = predict_by_chars(second)
                output += prediction
                if got_lucky:
                    hits += 1
        output.append('_')

    # Print out our input and the predicted sequence of keypresses
    print('%s\t%s' % (row[0], ' '.join(output)))

print('Hits:', hits, '; Tokens:', n_tokens, file=sys.stderr)
