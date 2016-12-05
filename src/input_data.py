# pylint: disable=missing-docstring
from collections import Counter

import time
import os
import numpy as np


def preprocess_input_files(data_dir, dataset):
    """ Preprocess the files to have only one sentence per
    infobox. This is to make sure that the learning example
    sees only the first sentence, because it is expected to
    generate the first sentence only."""
    in_sent = os.path.join(data_dir, dataset, '%s.sent' % (dataset))
    out_sent = os.path.join(data_dir, dataset, '%s_in.sent' % (dataset))
    nb_file = os.path.join(data_dir, dataset, '%s.nb' % (dataset))

    if os.path.isfile(out_sent):
        return

    with open(in_sent, 'r') as in_f,\
         open(out_sent, 'w') as out_f, open(nb_file, 'r') as nb_f:
        while True:
            first = in_f.readline()
            out_f.write(first)
            num_written = 1

            # Reached the end of sentences
            if not first:
                break

            # The number of lines from the same biography
            num_sent = int(nb_f.readline().rstrip())
            # Move the file pointer to the next infobox
            while num_written < num_sent:
                in_f.readline()
                num_written += 1


def create_dataset(data_dir, dataset, n, batch_size):
    """ create_dataset: Generate batch_sized examples

    Split sentences to chunks so that they can fit in within
    a same batch. Store the index to the corresponding table.
    The procedure distributes the sentence across multiple batches
    which can later be picked up by the train/valid procedure.

    Example:
    create_dataset('../data', 'train', 11, 32)

    Creates:
    ../data/train/train_x and ../data/train/train_y
    train_x contains sentences with length <= batch_size + (n - 1)
    train_y contains corresponding table number (to be indexed from
    ../data/train/train.box)
    """

    in_sent = os.path.join(data_dir, dataset, '%s_in.sent' % (dataset))

    # Read all the training sentences
    with open(in_sent) as sent_file:
        sentences = sent_file.readlines()

    table_num = 0

    x_path = os.path.join(data_dir, dataset, '%s_x' % (dataset))
    y_path = os.path.join(data_dir, dataset, '%s_y' % (dataset))

    with open(x_path, 'w') as data_x, \
         open(y_path, 'w') as data_y:
        for sentence in sentences:
            words = ['START'] * (n - 1)
            words.extend(sentence.split())

            while len(words) >= batch_size + n:
                # One batch includes batch_size + (n - 1) words
                # Form a sentence with appropriate number of words
                data_x.write(' '.join(words[:batch_size + (n - 1)]) + '\n')
                # Note down the corresponding table number
                data_y.write(str(table_num) + '\n')
                words = words[batch_size:]

            if len(words) != 0:
                # Leftover words go into a new batch
                data_x.write(' '.join(words) + '\n')
                data_y.write(str(table_num) + '\n')

            # Done processing words from the sentence.
            # Move to the next table, sentence
            table_num += 1


def get_hpca_embeddings(embed_path, d, k=20000):
    """get_hpca_embeddings : Return HPCA embeddings for top_k words

    Args:
        embed_path : Path to folder containing pre-computed
                     embeddings.
        d          : Dimension of the embeddings.
        k          : Size of the vocabulary.

    Returns:
        A numpy array embed containing embeddings for
        top_k words.
        Shape: (k, d)
    """

    embed_file = os.path.join(embed_path, 'embeddings.txt')
    embeddings = np.genfromtxt(embed_file)
    # Start and unknown embeddings
    unk_start = np.random.random((2, d))

    embed = np.vstack((unk_start, embeddings))
    hpca_embed_k = embed[:k]

    return hpca_embed_k


def w_index(embed_path, k):
    """w_index : Create word index for top_k words

    Args:
        embed_path : Path to folder containing top_k words.
        k          : Size of the vocabulary.

    Returns:
        word to index map (word2idx)
    """

    vocab_file = os.path.join(embed_path, 'vocabulary.txt')
    with open(vocab_file) as vocab:
        freq_words = vocab.readlines()

    # freq_words is a list of (words, freq) pairs
    # collect the words first
    words = [pair.strip().split()[0] for pair in freq_words]
    # append first two words in the index
    words = ['<unk>', '<start>'] + words[:k-2]

    # Create the forward index starting from 0
    word2idx = dict()
    for word in words:
        word2idx[word] = len(word2idx)

    print "Done creating w_index"
    return word2idx


def qword_vocab(data_dir, dataset, k):
    """ qword_vocab : Build an index for words appearing
    in the infobox.

    Args:
        data_dir : Path to the dataset.
        dataset  : Dataset in consideration.
        k        : Number of top_k words to be considered.

    Returns:
        Forward index into words in the infobox.
        (qword2idx)
    """
    print "Creating table word index"
    start = time.time()
    qwords = []

    ibox_path = os.path.join(data_dir, dataset, '%s.box' % (dataset))
    with open(ibox_path) as iboxs:
        tokens = iboxs.read().split()

    for token in tokens:
        (_, qword) = token.split(':', 1)
        if qword != '<none>':
            qwords.append(qword)

    counter = [['<unk>', '0']]
    counter.extend([list(item) for item in Counter(qwords).most_common(k-1)])

    qword2idx = dict()
    for qword, _ in counter:
        qword2idx[qword] = len(qword2idx)

    # Asserting the size of qword index
    assert max(qword2idx.values()) == k - 1

    duration = time.time() - start
    print "Created the table word index in %0.3f s" % (duration)
    return qword2idx


def field_name(field):
    """field_name : Extract the field name and position within
                    field from 'field'

    Args:
        field : field input as tokenized

    Returns:
        field_name : name of the field
        position : index of the digit(return 0 if no digit)

    Eg: name_first_1 returns name_first, 11
        teams_2 returns teams, 6
        name_first, returns name_first, 0
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
            return field[:pos], pos + 1
        else:
            # Two possibilities
            # either we have the last '_'
            # or we have to find the next '_'
            if '_' in field[pos+1:]:
                pos = field.index('_', pos + 1)
            else:
                return field, 0


def field_vocab(data_dir, dataset, min_freq):
    """ field_vocab : Generate the field vocabulary index

    Args:
        data_dir : Path to the data directory.
        dataset  : Dataset from which we build the field
                   index.
        min_freq : Minimum frequency of occurance of fields
                   to be indexed.

    Returns:
        field2idx (forward index for fields appearing
        atleast min_freq many times in dataset.)
        field_vocab_size : Size of the field vocab. (nF)
    """
    print "Creating field indexes"
    ibox_path = os.path.join(data_dir, dataset, '%s.box' % (dataset))

    start = time.time()
    fields = []
    # Collect the (field, value) pairs
    # across all infoboxes
    with open(ibox_path) as iboxs:
        tokens = iboxs.read().split()

    # Collect the list of field names
    # appearing across infoboxes
    for token in tokens:
        (field, _) = token.split(':', 1)
        (name, _) = field_name(field)
        fields.append(name)

    counter = [['<unk>', 0]]
    # Add field names of those fields that appear
    # atleast min_freq many times
    counter.extend([list(item) for item in Counter(fields).most_common()
                    if item[1] >= min_freq])
    field_vocab_size = len(counter)

    field2idx = dict()
    for field, _ in counter:
        field2idx[field] = len(field2idx)

    duration = time.time() - start

    print "Field vocabulary size : %d" % (field_vocab_size)
    print "Processed fields in %0.3f s" % (duration)

    return field2idx, field_vocab_size


def table_idx(table):
    """table_idx : Build an index for the table.
    Creates a dictionary with table words as keys
    and values as corresponding (field, start, end).

    Args:
        table : The processed table as given in the
                dataset.

    Returns:
        tableidx - a dictionary mapping table words to
                   their field name and position within
                   the field.
    """
    # Dictionary to count the number of words
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
                start = int(field[pos:])
                end = (count[name] - start) + 1
                if value in tableidx:
                    tableidx[value].append((name, start, end))
                else:
                    tableidx[value] = [(name, start, end)]
            else:
                if value in tableidx:
                    tableidx[value].append((name, 1, 1))
                else:
                    tableidx[value] = [(name, 1, 1)]
    return tableidx


def table_words(table):
    """ table_words - Return a list of unique words
    appearing in the table

    Args:
        table : The infobox

    Returns:
        List of unique words appearing in the table
    """
    tokens = table.split()
    twords = []

    for token in tokens:
        (_, t_word) = token.split(':', 1)
        if t_word != '<none>':
            twords.append(t_word)

    # Collect only the unique words appearing in the table
    words = list(set(twords))
    return words


def resize_index(word2idx, t_words):
    """ resize_index : Resize the index to account for words
    that appear in the table

    Args:
        word2idx    : The index of words in the vocabulary
        t_words : List of words appearing in the table

    Returns:
        Combined index of words from vocabulary and the table
        (wq2idx and idx2wq - Forward and reverse dictionaries)
    """
    # Collect the words in the vocabulary
    # and words outside the vocabulary that
    # are present in the table
    ws = word2idx.keys()
    out = set(t_words) - set(ws)

    # Sort the words to ensure that same order of
    # words is preserved on every call to resize_index
    out = list(out)
    out.sort()

    # Use copy to prevent modifying the original index
    wq2idx = word2idx.copy()

    # Extend the index with words from outside the vocabulary
    for word in out:
        wq2idx[word] = len(wq2idx)

    idx2wq = dict(zip(wq2idx.values(), wq2idx.keys()))

    # Sanity check
    assert len(wq2idx) == len(ws) + len(out)

    return wq2idx, idx2wq


def project_copy_scores(max_table_words, nW, wq2idx, t_words):
    """ project_copy_scores : Return a matrix to project the copy
    action scores to the output distribution.

    Args:
        max_table_words : Max. number of words in any table.
        nW              : Vocabulary size.
        wq2idx          : index from words to combined vocabulary.
        t_words         : list of words in the table.

    Return:
        A (num_words_in_table, nW + max_table_words) numpy array
    to project copy scores to the output distribution.
    """
    num_words_in_table = len(t_words)
    q_proj = np.zeros([num_words_in_table, (nW + max_table_words)],
                      dtype=float)

    # For each word in the table set the corresponding
    # position in the output position to 1
    for i in range(num_words_in_table):
        word = t_words[i]
        q_proj[i][wq2idx[word]] = 1

    return q_proj


def get_max_words_in_table(data_dir, dataset):
    """ get_max_words_in_table : Return the maximum number of words
    in any table in the dataset.

    Args:
        data_dir : Path to the data directory.
        dataset  : The dataset in consideration.

    Returns:
        Max. number of words appearing in any table in the dataset.
    """
    ibox_path = os.path.join(data_dir, dataset, '%s.box' % (dataset))

    with open(ibox_path) as ibox_f:
        iboxes = ibox_f.readlines()

    ws = [len(table_words(ibox)) for ibox in iboxes]
    max_words_in_table = max(ws)
    return max_words_in_table

<<<<<<< HEAD
=======
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
	
	idx2wq = dict(zip(wq2idx.values(), wq2idx.keys()))

	assert(len(wq2idx) == len(ws) + len(out))
	return wq2idx, idx2wq

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
>>>>>>> 4f7338cdf0872c6e86bb9620f504fce2f7bf700c

def getcopyaction(table, word_max_fields, field2idx):
    """ get_copy_action : Get the copy action embedding
    lookup.

    Args:
        table           : The infobox in consideration.
        word_max_fields : Max. number of fields in which
                          a word appears.
        field2idx       : Index for the fields.

    Returns:
        A lookup into the copy-action embeddings matrix.
        The matrix is [num_words_in_table, word_max_fields] sized.
    """
    # Collect the table words and the index
    # for the table.
    twords = table_words(table)
    tableidx = table_idx(table)

    copy_action = []
    for tword in twords:
        tw = []
        # Copy action lookup done using (field, start)
        # positions.
        for (field, pos, _) in tableidx[tword]:
            if field in field2idx:
                tw.append(field2idx[field] + pos)
            else:
                tw.append(field2idx['<unk>'] + pos)
        # Make sure that the number of entries
        # is equal to word_max_fields.
        if len(tw) < word_max_fields:
            tw.extend([tw[0]] * (word_max_fields - len(tw)))
        if len(tw) > word_max_fields:
            tw = tw[:word_max_fields]
        # Sanity check
        assert len(tw) == word_max_fields
        copy_action.append(tw)
    return copy_action


def global_context(table, max_fields, max_words, field2idx, qword2idx):
    """ global_context : Get the global context lookup entries.

    Args:
        table : The infobox in consideration.
        max_fields : Max fields in an infobox.
        max_words  : Max. words in an infobox.
        field2idx  : index for field names.
        qword2idx  : index for table words.

    Returns:
        gw : Global word conditioning lookup.
        gf : Global field conditioning lookup.
    """
    # Collect the (field, word) pairs in the infobox.
    tokens = table.split()
    gf = []
    gw = []

    for token in tokens:
        field, qword = token.split(':', 1)

        # If a valid table word
        if qword != '<none>':
            (name, _) = field_name(field)
            if name in field2idx:
                gf.append(field2idx[name])
            else:
                gf.append(field2idx['<unk>'])

            if qword in qword2idx:
                gw.append(qword2idx[qword])
            else:
                gw.append(qword2idx['<unk>'])

    # Same as with local conditioning we can encounter 2
    # scenarios.
    # 1. gw, gf smaller than expected
    # 2. gw, gf larger than expected

    # Case 1:
    if len(gf) < max_fields:
        gf.extend((max_fields - len(gf)) * [gf[0]])
    if len(gw) < max_words:
        gw.extend((max_words - len(gw)) * [gw[0]])

    # Case 2:
    if len(gf) > max_fields:
        gf = gf[:max_fields]
    if len(gw) > max_words:
        gw = gw[:max_words]

    # Sanity checks at the end
    assert len(gf) == max_fields
    assert len(gw) == max_words

    return gf, gw


def local_context(context, table, l, field2idx, word_max_fields):
    """ local_context : Generate the local context lookup given
    the input context.

    Args:
        context         : The current word context.
        table           : The infobox in consideration.
        l               : Max. number of words per field.
        field2idx       : Index to field names.
        word_max_fields : Max. number of fields a word appears in.

    Return:
        z_plus  : Lookup into the embeddings from start of field.
        z_minus : Lookup into the embeddings from end of field.
    """
    # Build the table index.
    tableidx = table_idx(table)
    z_plus = []
    z_minus = []

    for word in context:
        plus = []
        minus = []

        # Check if the current context word is in the
        # infobox.
        if word in tableidx:
            # Collect the list of occurances of the word.
            # across fields.
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
                    pos = field2idx['<unk>']

                # start, end is indexed from 1.
                # But the embedding matrices are indexed
                # from zeros. Adjust for this by subtracting 1.
                plus.append(pos + start - 1)
                minus.append(pos + end - 1)

        # Word in not present in the table values
        else:
            pos = field2idx['<unk>']
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
            plus.extend((word_max_fields - len(plus)) * [plus[0]])

        if len(minus) < word_max_fields:
            minus.extend((word_max_fields - len(minus)) * [minus[0]])

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


def setup(data_dir, embed_dir, n, batch_size, nW, min_field_freq, nQ):
<<<<<<< HEAD
    """ setup : Setup all the indexes and related startup functions.

    Args:
        data_dir       : Path to the data folder.
        embed_dir      : Path to the embeddings.
        batch_size     : Batch size for SGD
        nW             : Vocabulary size.
        min_field_freq : Min. freq. for field to be indexed.

    Returns:
        word2idx           : Word vocabulary index.
        field2idx          : Field index
        qword2idx          : Index to table words.
        nF                 : Size of the field index.
        max_words_in_table : Max. words in any table in the dataset.
    """
    preprocess_input_files('../data/', 'train')
    preprocess_input_files('../data/', 'test')
    preprocess_input_files('../data/', 'valid')

    train_x = os.path.join(data_dir, 'train', 'train_x')
    if not os.path.isfile(train_x):
        create_dataset(data_dir, 'train', n, batch_size)

    # Created just to maintain consistency with the DataSet
    # class. Not used in the learning procedure.
    test_x = os.path.join(data_dir, 'test', 'test_x')
    if not os.path.isfile(test_x):
        create_dataset(data_dir, 'test', n, batch_size)

    valid_x = os.path.join(data_dir, 'valid', 'valid_x')
    if not os.path.isfile(valid_x):
        create_dataset(data_dir, 'valid', n, batch_size)

    word2idx = w_index(embed_dir, nW)
    field2idx, nF = field_vocab(data_dir, 'train', min_field_freq)
    qword2idx = qword_vocab(data_dir, 'train', nQ)

    max_words_in_table = get_max_words_in_table(data_dir, 'train')

    return word2idx, field2idx, qword2idx, nF, max_words_in_table


class DataSet(object):
    """ DataSet : Class for defining the dataset as an object.

    """
    def __init__(self, data_dir, dataset, n, nW, nF, nQ, l, batch_size,
                 word2idx, field2idx, qword2idx, max_words, max_fields,
                 word_max_fields, max_words_in_table):
        self._dataset = dataset
        self._batch_size = batch_size
        self._n = n
        self._l = l
        self._nW = nW
        self._nF = nF
        self._nQ = nQ

        # Set up the indexes
        self._word2idx = word2idx
        self._field2idx = field2idx
        self._qword2idx = qword2idx

        # Parameters for global and local contexts
        # Max. words in an infobox
        self._max_words = max_words
        # Max. fields in an infobox
        self._max_fields = max_fields
        # Max. words per field
        self._word_max_fields = word_max_fields
        # Max. words in any table in the 'train' dataset.
        self._max_words_in_table = max_words_in_table

        # Infoboxes
        ibox_path = os.path.join(data_dir, dataset, '%s.box' % (dataset))
        with open(ibox_path) as iboxs:
            self._tables = iboxs.readlines()

        # x, y pairs
        # Load the x, y pairs as formed in the create_dataset
        # procedure.
        x_path = os.path.join(data_dir, dataset, '%s_x' % (dataset))
        y_path = os.path.join(data_dir, dataset, '%s_y' % (dataset))
        with open(x_path) as X, open(y_path) as Y:
            xs = X.readlines()
            ys = Y.readlines()

        self._xs = [x.rstrip() for x in xs]
        self._ys = [int(y.rstrip()) for y in ys]

        self._num_examples = len(self._xs)
        # Order of processing in a given epoch.
        self._sequence = np.arange(self._num_examples)

        self._context = ['<start>'] * (n - 1)

    def generate_permutation(self):
        """ Generate a permuation of the input
        examples.
        """
        np.random.shuffle(self._sequence)

    def num_examples(self):
        """ Number of examples in the dataset.
        """
        return self._num_examples

    def reset_context(self):
        """ Reset the context. Used while
        generating sentences.
        """
        self._context = ['<start>'] * (self._n - 1)

    def num_infoboxes(self):
        """ Number of infoboxes in the dataset.
        """
        return len(self._tables)

    def next_batch(self, pos):
        """ next_batch : Generate the next batch
        for training/validating the model.

        Args:
            pos : The current example being considered.
            pos varies from 1 - num_examples.
            The examples are processed based on the order
            generated by the random shuffle.
        """
        idx = self._sequence[pos]
        sentence = self._xs[idx]
        table_num = self._ys[idx]
        table = self._tables[table_num]
        words = sentence.split()

        contexts = []
        labels = []
        ct = []
        next_word = []
        z_plus = []
        z_minus = []

        twords = table_words(table)
        wq2idx, _ = resize_index(self._word2idx, twords)
        q_proj = project_copy_scores(self._max_words_in_table, self._nW,
                                     wq2idx, twords)

        # Generate contexts by unrolling n words
        # and sliding the window one word at a time.
        while len(words) >= self._n:
            contexts.append(words[:self._n-1])
            next_word.append(words[self._n-1])
            words = words[1:]

        # If we do not have enough contexts to fill in a
        # batch, repeat the last example to fill up the batch.
        if len(contexts) < self._batch_size:
            contexts.extend([contexts[-1]] *
                            (self._batch_size - len(contexts)))
            next_word.extend([next_word[-1]] *
                             (self._batch_size - len(next_word)))

        # Map words in the context to the vocabulary position.
        for context in contexts:
            ctxt = []
            for word in context:
                if word in self._word2idx:
                    ctxt.append(self._word2idx[word])
                else:
                    ctxt.append(self._word2idx['<unk>'])
            ct.append(ctxt)

        # Map the target words into words from the output
        # vocabulary. These can be outside the word2idx as
        # well.
        for word in next_word:
            if word in wq2idx:
                labels.append(wq2idx[word])
            else:
                labels.append(wq2idx['<unk>'])

        # Compute the local contexts for each context.
        for context in contexts:
            zp, zm = local_context(context, table, self._l,
                                   self._field2idx, self._word_max_fields)
            z_plus.append(zp)
            z_minus.append(zm)

        # Get the global context for the given table.
        gf, gw = global_context(table, self._max_fields, self._max_words,
                                self._field2idx, self._qword2idx)

        # Make batch_size many copies of the global
        # conditioning entries.
        global_field = [gf] * self._batch_size
        global_word = [gw] * self._batch_size
        copy = getcopyaction(table, self._word_max_fields, self._field2idx)

        return ct, z_plus, z_minus, global_field, global_word, \
               labels, copy, q_proj

    def next_single(self, pos, prev):
        """ next_single : Generate the next context for test/validation

        Args:
            pos  : The current table being considered.
            prev : The prediction from the previous step.

        Returns:
            context for input to predict the next word.
        """
        # Read the table
        table = self._tables[pos]
        twords = table_words(table)
        wq2idx, idx2wq = resize_index(self._word2idx, twords)
        q_proj = project_copy_scores(self._max_words_in_table,
                                     self._nW, wq2idx, twords)
        copy = getcopyaction(table, self._word_max_fields, self._field2idx)

        # Update the context.
        if prev in idx2wq:
            self._context = self._context[1:] + [idx2wq[prev]]
        else:
            self._context = self._context[1:] + ['<unk>']

        ct = []
        # Create the lookup for context.
        for word in self._context:
            if word in self._word2idx:
                ct.append(self._word2idx[word])
            else:
                ct.append(self._word2idx['<unk>'])

        # Generate the local conditioning variables.
        zp, zm = local_context(self._context, table, self._l,
                               self._field2idx, self._word_max_fields)

        # Generate the global conditioning lookups.
        gf, gw = global_context(table, self._max_fields, self._max_words,
                                self._field2idx, self._qword2idx)
        # next_word - Dummy position.
        # not used.
        next_word = self._word2idx['<unk>']
        return ct, zp, zm, gf, gw, next_word, copy, q_proj, idx2wq
=======
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

	test_x = os.path.join(data_dir, 'test', 'test_x')
	if not os.path.isfile(test_x):
		create_dataset(data_dir,'test', n, batch_size) 

	valid_x = os.path.join(data_dir, 'valid', 'valid_x')
	if not os.path.isfile(valid_x):
		create_dataset(data_dir,'valid', n, batch_size) 
	
	wpath = os.path.join(data_dir, 'train', 'train_in.sent')
	#word2idx, idx2word = build_vocab(wpath, nW, '../index')
	_, word2idx, idx2word = gethpcaembeddings(embed_dir, nW) 
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

	def num_infoboxes(self):
		return len(self._tables)

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
		wq2idx, _ = resize_index(self._word2idx, tablewords)
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

	def next_single(self, pos, previous_pred):
		table = self._tables[pos]
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
		wq2idx, idx2wq = resize_index(self._word2idx, tablewords)
		projection_mat = project_copy_scores(self._max_words_in_table, self._nW, wq2idx, tablewords)
		copy = getcopyaction(table, self._word_max_fields, self._field2idx)

		zp, zm = local_context(self._context, table, self._l, self._field2idx, self._word_max_fields)

		gf, gw = global_context(table, self._max_fields, self._max_words, self._field2idx, self._qword2idx)
		next_word = self._word2idx['UNK']
		return ct, zp, zm, gf, gw, next_word, copy, projection_mat, idx2wq
>>>>>>> 4f7338cdf0872c6e86bb9620f504fce2f7bf700c
