import sys
import pickle

# This is the training data
training_file = sys.argv[1]
# This is the filename that we will save the model in
model_file = sys.argv[2]

# This is a list of sentences in the training data
corpus = []

# Unigram counts = how often we see tokens, e.g. a frequency list
unigram_counts = {}
# Probability of the individual unigram
unigrams = {}
# How often we see word2 after word1 (e.g. 'house' after 'big')
half_bigram_counts = {}
half_bigrams = {}

bigram_counts = {}
bigrams = {}

# Read in the examples from the training data
for line in open(training_file).readlines():
    # Each line in the training data is two columns
    # тыкгаткэта ивнин риӄукэтэ	ты>кгат>кэ>та ив>ни>н риӄукэ>тэ
    row = line.strip().split('\t')
    # The tokens are in the first column
    tokens = row[0].split(' ')
    # Add the tokens to the list of sentences, with a beginning of sentence
    # and an end of sentence marker
    corpus.append(['#'] + tokens + ['#'])

# Collect unigram counts
n_tokens = 0  # Number of tokens
# For each of the sentences in our corpus
for sent in corpus:
    # For each of the tokens in the sentence
    for token in sent:
        # If we haven't seen the token before:
        if token not in unigram_counts:
            # Initialise the token count to zero
            unigram_counts[token] = 0
        # Increment the count of that token
        unigram_counts[token] += 1
    # Increment the count of all the tokens
    n_tokens += 1

# Estimate unigram probabilities
# For each of the types we have seen
for token in unigram_counts:
    # The probability is the frequency of the token divided by the total
    # number of tokens
    unigrams[token] = unigram_counts[token] / n_tokens

# Collect bigram counts
# e.g. ['the', 'big', 'house']
# bigrams: (the, big) (big, house)
# For each of the sentences in the corpus

'''half_bigrams = {
	'0': {}
	'1': {
		'первое слово в биграмме': {
			'1 символ 2ого слова в биграмме': 'частота такого продолжения'
		}
	},
	'2': '',
	...
	'длина самого большого 2торого слова - 2 символа': ''
}'''

half_bigram_counts['0'] = {}

for sent in corpus:
    for i in range(0, len(sent) - 1):
        w1 = sent[i]
        w2 = sent[i + 1]
        if w1 not in half_bigram_counts['0']:
            half_bigram_counts['0'][w1] = {}
        if w2 not in half_bigram_counts['0'][w1]:
            half_bigram_counts['0'][w1][w2] = 0
        half_bigram_counts['0'][w1][w2] += 1

        len_w2 = len(w2)
        for n in range(len_w2 - 2):
            if str(n + 1) not in half_bigram_counts:
                half_bigram_counts[str(n + 1)] = {}
            if w1 not in half_bigram_counts[str(n + 1)]:
                half_bigram_counts[str(n + 1)][w1] = {}
            n_w2 = w2[:n + 1]
            if n_w2 not in half_bigram_counts[str(n + 1)][w1]:
                half_bigram_counts[str(n + 1)][w1][n_w2] = 0
            half_bigram_counts[str(n + 1)][w1][n_w2] += 1

n_half_bigrams = 0

for key in half_bigram_counts.keys():
    half_bigrams[key] = dict()
    for token1 in half_bigram_counts[key]:
        if token1 not in half_bigrams[key]:
            half_bigrams[key][token1] = {}
        token_total = sum(half_bigram_counts[key][token1].values())
        for token2 in half_bigram_counts[key][token1]:
            if token2 not in half_bigrams[key][token1]:
                half_bigrams[key][token1][token2] = 0
                n_half_bigrams += 1
            half_bigrams[key][token1][token2] = half_bigram_counts[key][token1][token2] / token_total

for sent in corpus:
    for i in range(0, len(sent) - 1):
        w1 = sent[i]
        w2 = sent[i + 1]
        len_w2 = len(w2)
        for n in range(len_w2 - 1):
            w2_begin = w2[:n + 1]
            w2_end = w2[n + 1:]
            if str(n + 1) not in bigram_counts:
                bigram_counts[str(n + 1)] = {}
            if w1 not in bigram_counts[str(n + 1)]:
                bigram_counts[str(n + 1)][w1] = {}
            if w2_begin not in bigram_counts[str(n + 1)][w1]:
                bigram_counts[str(n + 1)][w1][w2_begin] = {}
            if w2_end not in bigram_counts[str(n + 1)][w1][w2_begin]:
                bigram_counts[str(n + 1)][w1][w2_begin][w2_end] = 0
            bigram_counts[str(n + 1)][w1][w2_begin][w2_end] += 1

n_bigrams = 0

for key in bigram_counts.keys():
    bigrams[key] = dict()

    for token1 in bigram_counts[key]:
        if token1 not in bigrams[key]:
            bigrams[key][token1] = {}
        for token2 in bigram_counts[key][token1]:
            if token2 not in bigrams[key][token1]:
                bigrams[key][token1][token2] = {}
            token_total = sum(bigram_counts[key][token1][token2].values())
            for token3 in bigram_counts[key][token1][token2]:
                if token3 not in bigrams[key][token1][token2]:
                    bigrams[key][token1][token2][token3] = 0
                    n_bigrams += 1
                bigrams[key][token1][token2][token3] = bigram_counts[key][token1][token2][token3] / token_total


# Write out model
mf = open(model_file, 'wb')
pickle.dump((unigrams, half_bigrams, bigrams), mf)

max_len = int(list(half_bigram_counts.keys())[-1])

# print(half_bigrams)
print('Written %d unigrams, %d half-bigrams and %d bigrams to %s.' %
      (len(unigrams.keys()), n_half_bigrams, n_bigrams, model_file))
