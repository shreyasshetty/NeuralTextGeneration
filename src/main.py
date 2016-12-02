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
from feed_dicts import placeholder_inputs_single
from feed_dicts import fill_feed_dict_single
from feed_dicts import do_eval

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
	
	# Generate the indexes
	word2idx, idx2word, field2idx, idx2field, nF, qword2idx, idx2qword, max_words_in_table = \
		setup(FLAGS.data_dir, '../embeddings', FLAGS.n, FLAGS.batch_size, FLAGS.nW, FLAGS.min_field_freq, FLAGS.nQ)

	# Create the dataset objects
	train_dataset = DataSet(FLAGS.data_dir,'train',FLAGS.n, FLAGS.nW, nF, \
							FLAGS.nQ, FLAGS.l, FLAGS.batch_size, word2idx, \
							idx2word, field2idx, idx2field, qword2idx, idx2qword, \
							FLAGS.max_words, FLAGS.max_fields, FLAGS.word_max_fields, max_words_in_table)
	num_train_examples = train_dataset.num_examples()
	
	valid_dataset =  DataSet(FLAGS.data_dir,'valid',FLAGS.n, FLAGS.nW, nF, \
							FLAGS.nQ, FLAGS.l, FLAGS.batch_size, word2idx, \
							idx2word, field2idx, idx2field, qword2idx, idx2qword, \
							FLAGS.max_words, FLAGS.max_fields, FLAGS.word_max_fields, max_words_in_table)
	num_valid_examples = valid_dataset.num_examples()

	test_dataset = DataSet(FLAGS.data_dir,'test',FLAGS.n, FLAGS.nW, nF, \
							FLAGS.nQ, FLAGS.l, FLAGS.batch_size, word2idx, \
							idx2word, field2idx, idx2field, qword2idx, idx2qword, \
							FLAGS.max_words, FLAGS.max_fields, FLAGS.word_max_fields, max_words_in_table)

	# The sizes of respective conditioning variables
	# for placeholder generation
	context_size = (FLAGS.n - 1)
	zp_size = context_size * FLAGS.word_max_fields
	zm_size = context_size * FLAGS.word_max_fields
	gf_size = FLAGS.max_fields
	gw_size = FLAGS.max_words
	copy_size = FLAGS.word_max_fields
	projection_size = FLAGS.nW + max_words_in_table

	# Generate the TensorFlow graph
	with tf.Graph().as_default():

		# Create the CopyAttention model
		model = CopyAttention(FLAGS.n, FLAGS.d, FLAGS.g, FLAGS.nhu, FLAGS.nW, nF, FLAGS.nQ, \
		                      FLAGS.nQpr, FLAGS.l, FLAGS.learning_rate, FLAGS.max_words, \
							  FLAGS.max_fields, FLAGS.word_max_fields, FLAGS.batch_size)	

		# Placeholders for train and validation
		context_pl, zp_pl, zm_pl, gf_pl, gw_pl, next_pl, copy_pl, projection_pl = \
			placeholder_inputs(FLAGS.batch_size, context_size, zp_size, zm_size, gf_size, gw_size, copy_size, projection_size)

		# Placeholders for test
		context_pl_t, zp_pl_t, zm_pl_t, gf_pl_t, gw_pl_t, copy_pl_t, projection_pl_t, next_pl_t = \
			placeholder_inputs_single(context_size, zp_size, zm_size, gf_size, gw_size, copy_size, projection_size)

		# Train and validation part of the model	
		predict = model.inference(FLAGS.batch_size, context_pl, zp_pl, zm_pl, gf_pl, gw_pl, copy_pl, projection_pl)
		loss = model.loss(predict, next_pl)
		train_op = model.train(loss)
		evaluate = model.evaluate(predict, next_pl)

		# Test component of the model
		pred_single = model.inference(1, context_pl_t, zp_pl_t, zm_pl_t, gf_pl_t, gw_pl_t, copy_pl_t, projection_pl_t)
		predicted_label = model.predicted_label(pred_single)
	
		# Initialize the variables and start the session	
		init = tf.initialize_all_variables()
		saver = tf.train.Saver()
		sess = tf.Session()
		sess.run(init)

		for epoch in range(1, FLAGS.num_epochs + 1):
			train_dataset.generate_permutation()		
			start_e = time.time()
			for i in range(num_train_examples):
				feed_dict = fill_feed_dict(train_dataset, i, context_pl, zp_pl, zm_pl, gf_pl, gw_pl, next_pl, copy_pl, projection_pl)	
				_, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

				if (i % FLAGS.print_every == 0):	
					print("Epoch : %d\tStep : %d\tLoss : %0.3f" %(epoch, i, loss_value))	

				if (i != 0 and i % FLAGS.valid_every == 0):
					print("Validation starting")
					valid_loss = do_eval(sess, predict, evaluate, valid_dataset, FLAGS.batch_size, context_pl, zp_pl, zm_pl, gf_pl, gw_pl, next_pl, copy_pl, projection_pl)
					print("Epoch : %d\tValidation loss: %0.5f" %(i, valid_loss))

				if (i != 0 and i % FLAGS.sample_every == 0):
					test_dataset.reset_context()
					pos = 0
					prev_predict = word2idx['START']
					while (pos != 1):
						with open(os.path.join(expt_result_path, 'results.txt'),'a') as exp:
							feed_dict_t = fill_feed_dict_single(test_dataset,prev_predict, 0, context_pl_t, zp_pl_t, zm_pl_t, gf_pl_t, gw_pl_t, next_pl_t, copy_pl_t, projection_pl_t)
							prev_predict = sess.run([predicted_label], feed_dict=feed_dict_t)
							prev = prev_predict[0][0][0]
							if prev in idx2word:
								exp.write(idx2word[prev] + ' ')
							else:
								exp.write('UNK ')
							if prev == word2idx['.']:
								pos = 1
								exp.write('\n')
							prev_predict = prev
	
			duration_e = time.time() - start

			print("Validation starting")
			start = time.time()
			valid_loss = do_eval(sess, predict, evaluate, valid_dataset, FLAGS.batch_size, context_pl, zp_pl, zm_pl, gf_pl, gw_pl, next_pl, copy_pl, projection_pl)
			duration = time.time() - start
			print("Epoch : %d\tValidation loss: %0.5f" %(i, valid_loss))
			print("Time taken for validating epoch %d : %0.3f" %(i, duration))
			with open(os.path.join(expt_result_path, str(i)+'_valid_loss'), 'w') as valid_loss_f:
				valid_loss_f.write("Epoch : %d\tValidation loss: %0.5f" %(i, valid_loss))

			checkpoint_file = os.path.join(chkpt_result_path, str(i) + '_checkpoint')
			saver.save(sess, checkpoint_file)

			print("Generating sentences for test dataset")
			start = time.time()
			num_test_boxes = test_dataset.num_infoboxes()
			test_sentences = os.path.join(expt_result_path, str(i) + '_sentences.txt')
			with open(test_sentences, 'a') as gen_sent:
				for k in range(num_test_boxes):
					pos = 0
					prev_predict = word2idx['START']
					test_dataset.reset_context()
					while(pos != 1):
						feed_dict_t = fill_feed_dict_single(test_dataset,prev_predict, k, context_pl_t, zp_pl_t, zm_pl_t, gf_pl_t, gw_pl_t, next_pl_t, copy_pl_t, projection_pl_t)
						prev_predict = sess.run([predicted_label], feed_dict=feed_dict_t)
						prev = prev_predict[0][0][0]
						if prev in idx2word:
							gen_sent.write(idx2word[prev] + ' ')
						else:
							gen_sent.write('UNK ')
						if prev == word2idx['.']:
							pos = 1
							gen_sent.write('\n')
						prev_predict = prev
			duration = time.time() - start
			print("Time taken to generate test sentences: %0.3f" %(duration))
			print("Time taken for epoch : %d is %0.3f minutes" %(epoch, duration_e/60))
	
if __name__ == "__main__":
	tf.app.run()
