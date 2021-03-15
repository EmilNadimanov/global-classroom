import sys, pickle

# This is the filename of the model file
model_file = sys.argv[1]
mf = open(model_file, 'rb')

# Load the unigram and bigram probabilities from the model file
(unigrams, bigrams, trigrams) = pickle.load(mf)

# Initialise the probability of the start of the sentence.
unigrams['#'] = 0.0

# Hits is number of times we get a prediction right
hits = 0
# The number of tokens
n_tokens = 0
# For each of the lines in the input
for line in sys.stdin.readlines():
	# Split into two columns
	row = line.strip().split('\t')
	# Our tokens are in column one, split by space
	tokens = row[0].split(' ')
	# The test tokens are the beginning of sentence symbol + the list of tokens
	tst_tokens = ['#'] + tokens
	# Increment the number of tokens by the length of the list containing the tokens
	n_tokens += len(tokens)
	
	# This is our output
	output = []
	# For each of the tokens in the "tst_tokens" list (e.g. the list + the beginning of sentence symbol)
	for i in range(len(tst_tokens)-2):
		first = tst_tokens[i] # First token in bigram
		second = tst_tokens[i+1] # Second token in bigram
		third = tst_tokens[i+2]
		# If we find the first token in the bigrams dict
		if first in trigrams:
			if second in trigrams[first]:
				pred = max(trigrams[first][second], key=trigrams[first][second].get)
				if pred == third:
					output.append(pred)
					hits += 1
				else:
					pred = max(bigrams[second], key=bigrams[second].get)
					if pred == third:
						output.append(pred)
						hits += 1
					else:
						output += [c for c in third]
			else:
				# попробую добавить максимально подходящую биграмму, затем - максимальную униграмму
				pred = max(bigrams[first], key=bigrams[first].get)
				pred = max(trigrams[first][pred], key=trigrams[first][pred].get)
				if pred == third:
					output.append(pred)
					hits += 1
				else:
					output += [c for c in third]
		else:
			pred = max(unigrams, key=unigrams.get)
			if pred == third:
				output.append(pred)
				hits += 1
			else:
				output += [c for c in third]
		output.append('_')

	# Print out our input and the predicted sequence of keypresses
	print('%s\t%s' % (row[0], ' '.join(output)))

print('Hits:', hits, '; Tokens:', n_tokens, file=sys.stderr)
