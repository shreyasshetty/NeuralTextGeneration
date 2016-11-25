from __future__ import division 

import time
import os
import json
import numpy as np
import tensorflow as tf

from pprint import pprint
from copyattention import CopyAttention
from input_data import DataSet
from input_data import setup

from feed_dicts import placeholder_inputs
from feed_dicts import fill_feed_dict
#from input_data import local_context
#from input_data import global_context

flags = tf.app.flags
# Model parameters
flags.DEFINE_integer("n", 11, "n-gram model parameter [11]")
flags.DEFINE_integer("d", 64, "Dimension of word embeddings [64]")
flags.DEFINE_integer("g", 128, "Dimension of global embedding [128]")
flags.DEFINE_integer("nhu", 256, "Number of hidden units [256]")
flags.DEFINE_integer("l", 10, "Max number of words per field [10]")
flags.DEFINE_float("learning_rate", 0.0025, "Learning rate parameter [0.0025]")

# Dataset related parameters
flags.DEFINE_integer("max_words", 100, "Maximum number of words in an infobox [100]")
flags.DEFINE_integer("max_fields", 10, "Maximum number of fields in an infobox [10]")
flags.DEFINE_integer("word_max_fields", 10, "Maximum of fields a word from an infobox can appear in [10]")
flags.DEFINE_integer("nW", 20000, "Size of the sentence vocabulary")
flags.DEFINE_integer("min_field_freq", 100, "Minimum frequency of occurance of a field [100]")

# Temporary flags - To be fixed
#flags.DEFINE_integer("nF", 1740, "Number of fields to be considered")
flags.DEFINE_integer("nQ", 20000, "Size of the table vocabulary")
flags.DEFINE_integer("nQpr", 1000, "Dummy")

# Experiment parameters
flags.DEFINE_integer("num_epochs", 10, "Number of epochs [10]")
flags.DEFINE_integer("batch_size", 32, "Batch size for SGD [32]")
flags.DEFINE_integer("print_every", 100, "Print out the training loss every #steps [100]")
flags.DEFINE_integer("sample_every", 1000, "Sample sentences every #steps [1000]")
flags.DEFINE_integer("test_every", 1000, "Test after every #steps [1000]")
flags.DEFINE_integer("valid_every", 1000, "Validate after every #steps [1000]")
flags.DEFINE_string("data_dir", '../data', "Path to the data directory [../data]")
flags.DEFINE_string("checkpoint_dir", '../checkpoint', "Directory to save checkpoints")
flags.DEFINE_string("experiment_dir", '../experiment', "Directory to store current experiment results")
FLAGS = flags.FLAGS	

#parser.add_option('--nQ', help='Num. words appearing a field values', dest='nQ', default=20000)
#parser.add_option('--nQpr', help='Num. words as modifications', dest='nQpr', default=11310)

def main(_):
	pprint(flags.FLAGS.__flags)	
	if not os.path.exists(FLAGS.experiment_dir):
		os.makedirs(FLAGS.experiment_dir)
		expt_num = "1"
	else:
		expt_num = str(max(map(int, os.listdir(FLAGS.experiment_dir))) + 1)
	expt_result_path = os.path.join(FLAGS.experiment_dir, expt_num)
	os.makedirs(expt_result_path)

	if not os.path.exists(FLAGS.checkpoint_dir):
		os.makedirs(FLAGS.checkpoint_dir)
	chkpt_result_path = os.path.join(FLAGS.checkpoint_dir, expt_num)
	os.makedirs(chkpt_result_path)

	with open(os.path.join(expt_result_path, "params.json"), 'w') as params_e, \
	     open(os.path.join(chkpt_result_path, "params.json"), 'w') as params_c:
		json.dump(flags.FLAGS.__flags, params_e)
		json.dump(flags.FLAGS.__flags, params_c)
	
	word2idx, idx2word, field2idx, idx2field, nF, qword2idx, idx2qword, max_words_in_table = \
		setup(FLAGS.data_dir, FLAGS.nW, FLAGS.min_field_freq, FLAGS.nQ)
	
	train_dataset = DataSet(FLAGS.data_dir,'train',FLAGS.n, FLAGS.nW, nF, \
							FLAGS.nQ, FLAGS.l, FLAGS.batch_size, word2idx, \
							idx2word, field2idx, idx2field, qword2idx, idx2qword, \
							FLAGS.max_words, FLAGS.max_fields, FLAGS.word_max_fields, max_words_in_table)

	context_size = (FLAGS.n - 1)
	zp_size = context_size * FLAGS.word_max_fields
	zm_size = context_size * FLAGS.word_max_fields
	gf_size = FLAGS.max_fields
	gw_size = FLAGS.max_words
	copy_size = FLAGS.word_max_fields
	projection_size = FLAGS.nW + max_words_in_table

	with tf.Graph().as_default():
		model = CopyAttention(FLAGS.n, FLAGS.d, FLAGS.g, FLAGS.nhu, FLAGS.nW, nF, FLAGS.nQ, \
		                      FLAGS.nQpr, FLAGS.l, FLAGS.learning_rate, FLAGS.max_words, \
							  FLAGS.max_fields, FLAGS.word_max_fields, FLAGS.batch_size)	

		context_pl, zp_pl, zm_pl, gf_pl, gw_pl, next_pl, copy_pl, projection_pl = \
			placeholder_inputs(FLAGS.batch_size, context_size, zp_size, zm_size, gf_size, gw_size, copy_size, projection_size)
		
		predict = model.inference(context_pl, zp_pl, zm_pl, gf_pl, gw_pl, copy_pl, projection_pl, FLAGS.batch_size)
		loss = model.loss(predict, next_pl)
		train_op = model.train(loss)

		init = tf.initialize_all_variables()
		sess = tf.Session()
		
		sess.run(init)

		num_examples = train_dataset.num_examples()
		
		for epoch in range(1, FLAGS.num_epochs + 1):
			train_dataset.generate_permuation()		
			start = time.time()
			for i in range(num_examples):
				feed_dict = fill_feed_dict(train_dataset, i, context_pl, zp_pl, zm_pl, gf_pl, gw_pl, next_pl)	
				_, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

				if (i % FLAGS.print_every == 0):	
					print("Epoch : %d\tStep : %d\tLoss : %0.3f" %(epoch, i, loss_value))	
	
			duration = time.time() - start
			print("Time taken for epoch : %d is %0.3f minutes" %(epoch, duration/60))
	
if __name__ == "__main__":
	tf.app.run()
