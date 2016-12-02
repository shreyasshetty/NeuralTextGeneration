import time
import os
import pickle
import numpy as np
from collections import Counter

def preprocess_input_files(dataset):
	""" Preprocess the files to have only one sentence per
	infobox. This is to make sure that the learning example
	sees only the first sentence, because it is expected to
	generate the first sentence only.
	
	sent_w : file pointer for the output sentence file
	sent_r : file pointer to the sentence files from the 
	 		   dataset
	nb_f   : file pointer to the number file."""
	
	#sent_in_path = '../data/%s/%s_in.sent' %(dataset,dataset)
	#sent_r_path = '../data/%s/%s.sent' %(dataset, dataset)
	#nb_f_path = '../data/%s/%s.nb' %(dataset, dataset)
	
	data_dir = '../data'

	sent_in_path = os.path.join(data_dir, dataset, '%s_in.sent' %(dataset))
	sent_r_path = os.path.join(data_dir, dataset, '%s.sent' %(dataset))
	nb_f_path = os.path.join(data_dir, dataset, '%s.nb' %(dataset))

	if os.path.isfile(sent_in_path):
		return

	with open(sent_in_path,'w') as sent_w:
		with open(sent_r_path,'r') as sent_r:
			with open(nb_f_path,'r') as nb_f:
				while True:
					train_sent = sent_r.readline()
					# If done reading all lines, break out of the loop
					if not train_sent:
						break
					num_sent = int(nb_f.readline().rstrip())
					train_w = sent_w.write(train_sent)
					num_written = 1
					while num_written < num_sent:
						sent_r.readline()
						num_written = num_written + 1

def create_dataset(data_dir, dataset, batch_size):
	"""
	Create a (x,y) pair of x as input sentence and
	y as the index of the table. The resultant
	(x,y) pairs have a property that len(x) <= batch_size.

	Eg: sentence : "This is the house of the prime minister"
	    table : corresponding_table (index 5)
		batch_size : 5
		
		(x, y) pairs:
		("This is the house of", 5)
		("the prime minister", 5)

		Writes the following into dataset_x and dataset_y respectively
	"""

	with open(os.path.join(data_dir,dataset,'%s_in.sent' %(dataset))) as sent_file:
		sentences = sent_file.readlines()

	num_sentences = len(sentences)

	current_idx = 0
	with open(os.path.join(data_dir, dataset, '%s_x' %(dataset)),'w') as data_x, \
		 open(os.path.join(data_dir, dataset, '%s_y' %(dataset)), 'w') as data_y:
		for sentence in sentences:
			words = sentence.split()
			if len(words) < batch_size:
				data_x.write(' '.join(words) + '\n')
				data_y.write(str(current_idx) + '\n')
				current_idx += 1
			else:
				while (len(words) > batch_size):
					data_x.write(' '.join(words[:batch_size])+ '\n')
					data_y.write(str(current_idx) + '\n')
					words = words[batch_size:]
				if len(words) != 0:
					data_x.write(' '.join(words) + '\n')
					data_y.write(str(current_idx) + '\n')
				current_idx += 1
	
def build_vocab(filename, k, idx_path):
	w2idx_path = os.path.join(idx_path, 'word2idx.p')
	idx2w_path = os.path.join(idx_path, 'idx2word.p')
	if os.path.isfile(w2idx_path) and os.path.isfile(idx2w_path):
		print("Reading previously stored word indexes")
		word2idx = pickle.load(open(w2idx_path, 'rb'))
		idx2word = pickle.load(open(idx2w_path, 'rb'))
		assert(max(word2idx.values()) == k - 1)
		return word2idx, idx2word
		
	start = time.time()
	with open(filename) as f:
		words = [word for line in f.readlines() for word in line.split()]
	
	total_count = len(words)
	counter = [['UNK', 0], ['START',1]]
	counter.extend([list(item) for item in Counter(words).most_common(k-2)])
	vocab_size = len(counter)

	# Quick test to see if we have collected the same number of words
	# as desired
	assert(vocab_size == k)
	
	word2idx = dict()
	for word, _ in counter:
		word2idx[word] = len(word2idx)
		
	assert('START' in word2idx)
		
	#word2idx['END'] = len(word2idx)
	#Assert that the values are 0 indexed with max index to num. of
	#words in vocabulary - 2
	#having k words in vocab equivalent to k classes
	# 0 indexed implies max. value = k - 1
	assert(max(word2idx.values()) == k - 1)

	idx2word = dict(zip(word2idx.values(), word2idx.keys()))
		#assert('START' in idx2word)
	duration = time.time() - start
	print("Building vocabulary for words in sentences")
	print("%d words processed in %0.5f seconds" %(total_count,duration)) 
	
	print("Pickling the word indexes")
	pickle.dump(word2idx, open(w2idx_path, 'wb'))
	pickle.dump(idx2word, open(idx2w_path, 'wb'))
	# TEST code:
	#max_class = max(word2idx.values())
	#print("Max num classes : %d" %(max_class))
	#min_class = min(word2idx.values())
	#print("Min class :%d" %(min_class))
	# END OF TEST code
	return word2idx, idx2word

def qword_vocab(filename, k, idx_path):
	q2idx_path = os.path.join(idx_path, 'qword2idx.p')
	idx2q_path = os.path.join(idx_path, 'idx2qword.p')

	if os.path.isfile(q2idx_path) and os.path.isfile(idx2q_path):
		print("Reading saved indexes")
		qword2idx = pickle.load(open(q2idx_path, 'rb'))
		idx2qword = pickle.load(open(idx2q_path, 'rb'))
		return qword2idx, idx2qword

	start = time.time()
	qwords = list()

	with open(filename) as f:
		tokens = f.read().split()

	for token in tokens:
		(field, qword) = token.split(':',1)	
		if qword != '<none>':
			qwords.append(qword)
	
	total_count = len(qwords)
	counter = [['UNK', '0']]
	counter.extend([list(item) for item in Counter(qwords).most_common(k-1)])
	vocab_size = len(counter)

	assert(vocab_size == k)
	qword2idx = dict()
	for qword, _ in counter:
		qword2idx[qword] = len(qword2idx)

	#Asserting the size of qword index
	assert(max(qword2idx.values()) == k - 1)

	idx2qword = dict(zip(qword2idx.values(), qword2idx.keys()))
	duration = time.time() - start
	print("Building vocabulary for words in infoboxes")
	print("%d words processed in %0.5f seconds" %(total_count, duration))

	print("Saving indexes")
	pickle.dump(qword2idx, open(q2idx_path, 'wb'))
	pickle.dump(idx2qword, open(idx2q_path, 'wb'))
	return qword2idx, idx2qword

def qprime_vocab(filename):
	""" Form a qprime vocabulary
	"""
	with open(filename) as qp:
		qprimewords = qp.readlines()

	clean = lambda x: x.rstrip()
	qprimewords = map(clean, qprimewords)
	#qprimewords = [word.rstrip() for word in qprimewords]
	counter = [['UNK', '0']]
	counter.extend([list(item) for item in Counter(qprimewords).most_common()])
	qprimeword2idx = dict()
	for qprimeword, _ in counter:
		qprimeword2idx[qprimeword] = len(qprimeword2idx)

	idx2qprimeword = dict(zip(qprimeword2idx.values(), qprimeword2idx.keys()))
	return qprimeword2idx, idx2qprimeword

def field_name(field):
	"""Extract field name from field
	Eg: name_first_1 return name_first
		teams_1, return teams and the position
		of the position i.e 1 (6 in this case)
		returns position to be 0 if the name
		of the field is the entire string 'field'
	"""
	if '_' not in field:
		return field, 0
	pos = field.index('_', 0)
	while pos != -1:
		
		# Added after observing that fields of type
		# nicknames_ exist in the dataset.
		if len(field) == (pos + 1):
			return field[:pos], 0

		if field[pos+1].isdigit():
			return field[:pos], pos
		else:
			# Two possibilities
			# either we have the list '_'
			# or we have to find the next '_'
			if '_' in field[pos+1:]:
				pos = field.index('_', pos + 1)
			else:
				return field, 0

def field_vocab(filename, min_freq, idx_path):
	field_idx_file = os.path.join(idx_path, 'f2idx.p')
	idx_field_file = os.path.join(idx_path, 'idx2f.p')
	
	if os.path.isfile(field_idx_file) and os.path.isfile(idx_field_file):
		print("Loading saved indexes")
		field2idx = pickle.load(open(field_idx_file, 'rb'))
		idx2field = pickle.load(open(idx_field_file, 'rb'))
		vocab_size = len(field2idx.keys())
		return field2idx, idx2field, vocab_size

	start = time.time()
	fields = list() 
	with open(filename) as f:
		tokens = f.read().split() 
	
	for token in tokens:
		(field,_) = token.split(':',1)
		(name, _) = field_name(field)
		fields.append(name)
	
	total_count = len(fields)
	counter = [['UNK', 0], ['START', 1]]
	# Previous approach.
	#counter.extend([list(item) for item in Counter(fields).most_common(k)])
	# We use the fields that appear atleast 1000 time in the training data
	counter.extend([list(item) for item in Counter(fields).most_common() if item[1] >= min_freq])
	vocab_size = len(counter)
		
	print("Field vocab size: %d" %(vocab_size))
	# Quick test to see if we have collected the same number of fields
	# as desired
	# assert(vocab_size == (k + 1))
		
	field2idx = dict()
	for field, _ in counter:
		field2idx[field] = len(field2idx)

	idx2field = dict(zip(field2idx.values(), field2idx.keys()))
	duration = time.time() - start
	print("Created field vocabulary")
	print("%d words processed in %0.5f seconds" %(total_count,duration)) 

	print("Saving the indexes")
	pickle.dump(field2idx, open(field_idx_file, 'wb'))
	pickle.dump(idx2field, open(idx_field_file, 'wb'))
	return field2idx, idx2field, vocab_size

def table_idx(table):
	""" Generate an index for the table.
	To be used by the local_context function.
	Returns a dictionary with the words in the 
	table as keys and values as (field, start, end)
	"""
	# Dictionary to count the number of values
	# for a given field
	count = dict()
	# The eventual index we return
	tableidx = dict()
	tokens = table.split()
	
	# Populate the count dictionary
	for token in tokens:
		field, value = token.split(':', 1)
		if value != '<none>':
			(name, _) = field_name(field)
			if name in count:
				count[name] += 1
			else:
				count[name] = 1

	# Now that we know number of tokens in a given
	# field, we go and fill the dictionary tableidx
	for token in tokens:
		field, value = token.split(':', 1)
		if value != '<none>':
			(name, pos) = field_name(field)
			if pos != 0:
				start = int(field[pos+1:])
				end = (count[name] - start) + 1
				if value in tableidx:
					tableidx[value].append((name, start, end))
				else:
					tableidx[value] = [(name, start, end)]
			else:
				if value in tableidx:
					tableidx[value].append((name,1,1))
				else:
					tableidx[value] = [(name, 1, 1)]
	return tableidx

def copy_action_inputs(table, tableidx, l, field2idx, word_max_fields):
	words_in_table = tableidx.keys() 
	c_in = []
	for w in words_in_table:
		positions = tableidx[w]
		c_w = []
		for position in positions:
			field_name = position[0]
			field_pos = position[1]
			if field_name in field2idx:
				c_w.append(field2idx[field_name]*l + field_pos)
			else:
				c_w.append(field2idx['UNK'])
		if len(c_w) >= word_max_fields:
			c_w = c_w[:word_max_fields]
		else:
			c_w = c_w + [c_w[0]]*(word_max_fields - len(c_w))
		assert(len(c_w) == word_max_fields)
		c_in.append(c_w)

	assert(len(c_in) == len(words_in_table))
	return c_in

def local_context(context, table, l, field2idx, word_max_fields):
	""" Generate the local context
		table is a collection of 
	"""
	tableidx = table_idx(table)	
	z_plus = list()
	z_minus = list()

	for word in context:
		plus = list()
		minus = list()
		if word in tableidx:
			fields = tableidx[word]	
			for field in fields:
				(name, start, end) = field

				# Account for the corner case the we might 
				# have some word beyond 'l' distance
				# We just ignore if that is the case
				if start > l or end > l:
					pass
				
				if name in field2idx:
					pos = field2idx[name]
				else:
					pos = field2idx['UNK']
				plus.append(pos + start - 1)
				minus.append(pos + end - 1)

		# Word in not present in the table values
		else:
			pos = field2idx['UNK']
			plus.append(pos)
			minus.append(pos)

		# After filling up the word indices, we could end
		# with two possibilities
		# 1. Fewer than word_max_fields indices
		# 2. Greater than word_max_fields indices

		# Case 1
		# Fix by filling up the remaining space by
		# repeating the first element
		if len(plus) < word_max_fields:
			plus.extend((word_max_fields - len(plus))*[plus[0]])
		
		if len(minus) < word_max_fields:
			minus.extend((word_max_fields - len(minus))*[minus[0]])

		# Case 2
		# Fix by cutting off the list to just have 
		# word_max_fields many entries
		if len(plus) > word_max_fields:
			plus = plus[:word_max_fields]
		if len(minus) > word_max_fields:
			minus = minus[:word_max_fields]

		z_plus.extend(plus)
		z_minus.extend(minus)
	
	return z_plus, z_minus

def global_context(table, max_fields, max_words, field2idx, qword2idx):
	tokens = table.split()
	gf = list()
	gw = list()
	
	for token in tokens:
		field, qword = token.split(':',1)
		
		# If we are in a valid field
		if qword != '<none>':			
			(name, _) = field_name(field)
			if name in field2idx:
				gf.append(field2idx[name])
			else:
				gf.append(field2idx['UNK'])

			if qword in qword2idx:
				gw.append(qword2idx[qword])
			else:
				gw.append(qword2idx['UNK'])
		
	# Same as with local conditioning we can encounter 2
	# scenarios.
	# 1. gw, gf smaller than expected
	# 2. gw, gf larger than expected
		
	# Case 1:
	if len(gf) < max_fields:
		gf.extend((max_fields - len(gf))*[gf[0]])
	if len(gw) < max_words:
		gw.extend((max_words - len(gw))*[gw[0]])

	# Case 2:
	if len(gf) > max_fields:
		gf = gf[:max_fields]
	if len(gw) > max_words:
		gw = gw[:max_words]

	# Sanity checks at the end
	assert(len(gf) == max_fields)
	assert(len(gw) == max_words)

	return gf, gw

def merge_vocab(word2idx, qword2idx):
	start = time.time()
	words = word2idx.keys()
	qwords = qword2idx.keys()
	words_set = set(words)
	qwords_set = set(qwords)
	words_set.update(qwords_set)
	vocab = list(words_set)
	vocab_size = len(vocab)	
	total = len(words) + len(qwords)

	key2idx = dict()
	for word in vocab:
		key2idx[word] = len(key2idx)
	
	idx2key = dict(zip(key2idx.values(), key2idx.keys()))
	duration = time.time() - start
	print("Merged the word vocab and table word vocab")
	print("New vocabulary size is %d" %(vocab_size))
	print("Processed %d words in %.5f seconds" %(total, duration))
	return key2idx, idx2key


def table_words(table):
	tokens = table.split()
	twords = []

	for token in tokens:
		(_,tWord) = token.split(':',1)
		if tWord != '<none>':
			twords.append(tWord)
	
	uniq = set(twords)
	words = list(uniq)
	return words

def resize_index(word2idx, tableWords):
	"""
	Create a combined index of words in
	W union Q (tableWords).
	"""
	ws = word2idx.keys()	
	out = set(tableWords) - set(ws)
	wq2idx = word2idx.copy()

	for word in out:
		wq2idx[word] = len(wq2idx)

	assert(len(wq2idx) == len(ws) + len(out))
	return wq2idx

def project_copy_scores(max_table_words, nW, wq2idx, tableWords):
	"""
	Return a (nW + max_table_words)*tableWords transformation matrix
	that projects copy scores into the output distribution.
	"""
	num_words_in_table = len(tableWords)
	q_proj = np.zeros([(nW + max_table_words), num_words_in_table], dtype=float)
	for i in range(num_words_in_table):
		word = tableWords[i]
		q_proj[wq2idx[word]][i] = 1
	return q_proj

def getmaxwordsintable(data_dir, dataset):
	with open(os.path.join(data_dir, dataset, '%s.box' %(dataset)), 'r') as box_f:
		boxes = box_f.readlines()
	
	ws = [len(table_words(box)) for box in boxes]
	max_words_in_table = max(ws)
	return max_words_in_table

def getcopyaction(table, word_max_fields, field2idx):
	tablewords = table_words(table)
	tableidx = table_idx(table)
	copy = []
	for tword in tablewords:
		tw = []
		for (field,pos,_) in tableidx[tword]:
			if field in field2idx:
				tw.append(field2idx[field] + pos)
			else:
				tw.append(field2idx['UNK'] + pos)
		if len(tw) < word_max_fields:
			tw.extend([tw[0]]* (word_max_fields - len(tw)))
		if len(tw) > word_max_fields:
			tw = tw[:word_max_fields]
		assert(len(tw) == word_max_fields)
		copy.append(tw)
	return copy

def create_dataset(data_dir, dataset, n, batch_size):
	"""	Create a (x,y) pair of x as input sentence and
	y as the index of the table. The resultant
	(x,y) pairs have a property that len(x) <= batch_size.

	Eg: sentence : "This is the house of the prime minister"
	    table : corresponding_table (index 5)
		batch_size : 5
		
		Returns:
			("This is the house of", 5)
			("the prime minister", 5)
	"""

	with open(os.path.join(data_dir,dataset,'%s_in.sent' %(dataset))) as sent_file:
		sentences = sent_file.readlines()

	num_sentences = len(sentences)

	current_idx = 0
	with open(os.path.join(data_dir, dataset, '%s_x' %(dataset)),'w') as data_x, \
		 open(os.path.join(data_dir, dataset, '%s_y' %(dataset)), 'w') as data_y:
		for sentence in sentences:
			words = ['START'] * (n - 1)
			words.extend(sentence.split())
			while (len(words) >= batch_size + n):
				data_x.write(' '.join(words[:batch_size + (n - 1)])+ '\n')
				data_y.write(str(current_idx) + '\n')
				words = words[batch_size:]
			if (len(words) != 0):
				data_x.write(' '.join(words) + '\n')
				data_y.write(str(current_idx) + '\n')
			current_idx += 1

def setup(data_dir, n, batch_size, nW, min_field_freq, nQ):
	""" A function which prepares all the indexes and other needed 
	preliminaries to run the program.
	Run once in main() in the train file.
	"""
	preprocess_input_files('train')
	preprocess_input_files('test')
	preprocess_input_files('valid')
	
	train_x = os.path.join(data_dir, 'train', 'train_x')
	if not os.path.isfile(train_x):
		create_dataset(data_dir,'train', n, batch_size) 

	valid_x = os.path.join(data_dir, 'valid', 'valid_x')
	if not os.path.isfile(valid_x):
		create_dataset(data_dir,'valid', n, batch_size) 
	
	wpath = os.path.join(data_dir, 'train', 'train_in.sent')
	word2idx, idx2word = build_vocab(wpath, nW, '../index')
	fqpath = os.path.join(data_dir, 'train', 'train.box')
	field2idx, idx2field, nF = field_vocab(fqpath, min_field_freq, '../index')
	qword2idx, idx2qword = qword_vocab(fqpath, nQ, '../index')
	
	max_words_in_table = getmaxwordsintable(data_dir, 'train')

	return word2idx, idx2word, field2idx, idx2field, nF, qword2idx, idx2qword, max_words_in_table


class DataSet(object):
	def __init__(self, data_dir, dataset, n, nW, nF, nQ, l, batch_size,	
	            word2idx, idx2word, field2idx, idx2field, 
				qword2idx, idx2qword, max_words, max_fields, word_max_fields, max_words_in_table):
		self._dataset = dataset
		self._batch_size = batch_size
		self._n = n
		self._l = l
		self._nW = nW
		self._nF = nF
		self._nQ = nQ

		# Set up the indexes
		self._word2idx = word2idx
		self._idx2word = idx2word
		self._field2idx = field2idx
		self._idx2field = idx2field
		self._qword2idx = qword2idx
		self._idx2qword = idx2qword
		
		# Parameters for global and local contexts
		self._max_words = max_words  # Max. words in an infobox
		self._max_fields = max_fields # Max. fields in an infobox
		self._word_max_fields = word_max_fields # Max. words per field
		self._max_words_in_table = max_words_in_table

		# Infoboxes
		with open(os.path.join(data_dir, dataset, '%s.box' %(dataset)), 'r') as table_f:
			self._tables = table_f.readlines()

		# x, y pairs
		with open(os.path.join(data_dir, dataset, '%s_x' %(dataset))) as X, \
			 open(os.path.join(data_dir, dataset, '%s_y' %(dataset))) as Y:
			xs = X.readlines()
			ys = Y.readlines()

		self._xs = map(lambda x: x.rstrip(), xs)
		self._ys = map(lambda x: int(x.rstrip()), ys)

		self._num_examples = len(self._xs)
		self._sequence = np.arange(self._num_examples)

		self._context = ['START'] * (self._n - 1)

	def generate_permutation(self):
		np.random.shuffle(self._sequence)

	def num_examples(self):
		return self._num_examples
	
	def reset_context(self):
		self._context = ['START'] * (self._n - 1)

	def next_batch(self, pos):
		idx = self._sequence[pos]

		# Sentence for the current example
		sentence = self._xs[idx]
		
		# index for the table in the current example
		tablepos = self._ys[idx]
		table = self._tables[tablepos]

		# 'START' tokens already appended
		words = sentence.split()
		contexts = []
		labels = []
		ct = []
		next_word = []
		z_plus = []
		z_minus = []

		tablewords = table_words(table)
		wq2idx = resize_index(self._word2idx, tablewords)
		copy_projection_matrix = project_copy_scores(self._max_words_in_table, self._nW, wq2idx, tablewords)

		while(len(words) >= self._n):
			contexts.append(words[:self._n-1])
			next_word.append(words[self._n-1])
			words = words[1:]

		assert(len(next_word) == len(contexts))

		if (len(contexts) < self._batch_size):
			contexts.extend([contexts[-1]] * (self._batch_size - len(contexts)))			
			next_word.extend([next_word[-1]] * (self._batch_size - len(next_word)))

		assert(len(contexts) == self._batch_size)
		assert(len(next_word) == self._batch_size)

		for context in contexts:
			ctxt = []
			for word in context:
				if word in self._word2idx:
					ctxt.append(self._word2idx[word])
				else:
					ctxt.append(self._word2idx['UNK'])
			ct.append(ctxt)

		for word in next_word:
			if word in wq2idx:
				labels.append(wq2idx[word])
			else:
				labels.append(wq2idx['UNK'])

		for context in contexts:
			zp, zm = local_context(context, table, self._l, self._field2idx, self._word_max_fields)
			z_plus.append(zp)
			z_minus.append(zm)

		gf, gw = global_context(table, self._max_fields, self._max_words, self._field2idx, self._qword2idx)
		global_field = [gf] * self._batch_size
		global_word = [gw] * self._batch_size
		copy = getcopyaction(table, self._word_max_fields, self._field2idx)

		return ct, z_plus, z_minus, global_field, global_word, labels, copy, copy_projection_matrix

	def next_single(self, previous_pred):
		table = self._tables[1]	
		if previous_pred in self._idx2word:
			self._context = self._context[1:] + [self._idx2word[previous_pred]]
		else:
			self._context = self._context[1:] + ['UNK']

		ct = []
		for word in self._context:
			if word in self._word2idx:
				ct.append(self._word2idx[word])
			else:
				ct.append(self._word2idx['UNK'])
		
		tablewords = table_words(table)
		wq2idx = resize_index(self._word2idx, tablewords)
		projection_mat = project_copy_scores(self._max_words_in_table, self._nW, wq2idx, tablewords)
		copy = getcopyaction(table, self._word_max_fields, self._field2idx)

		zp, zm = local_context(self._context, table, self._l, self._field2idx, self._word_max_fields)

		gf, gw = global_context(table, self._max_fields, self._max_words, self._field2idx, self._qword2idx)
		next_word = self._word2idx['UNK']
		return ct, zp, zm, gf, gw, next_word, copy, projection_mat
