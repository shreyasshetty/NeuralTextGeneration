from __future__ import division

import time
import os
import json
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
flags.DEFINE_integer("max_fields", 10, "Maximum number of fields in an infobox [10]")
flags.DEFINE_integer("word_max_fields", 10, "Maximum of fields a word from an infobox can appear in [10]")
flags.DEFINE_integer("nW", 20000, "Size of the sentence vocabulary")
flags.DEFINE_integer("min_field_freq", 100, "Minimum frequency of occurance of a field [100]")

# Temporary flags - To be fixed
flags.DEFINE_integer("nQ", 20000, "Size of the table vocabulary")
flags.DEFINE_integer("nQpr", 1000, "Dummy")

# Experiment parameters
flags.DEFINE_integer("num_epochs", 10, "Number of epochs [10]")
flags.DEFINE_integer("batch_size", 32, "Batch size for SGD [32]")
flags.DEFINE_string("xavier", "True", "Initialize using Xavier initialization[True]")
flags.DEFINE_integer("print_every", 100, "Print out the training loss every #steps [100]")
flags.DEFINE_integer("sample_every", 1000, "Sample sentences every #steps [1000]")
flags.DEFINE_integer("test_every", 1000, "Test after every #steps [1000]")
flags.DEFINE_integer("valid_every", 1000, "Validate after every #steps [1000]")
flags.DEFINE_string("data_dir", '../data', "Path to the data directory [../data]")
flags.DEFINE_string("checkpoint_dir", '../checkpoint', "Directory to save checkpoints")
flags.DEFINE_string("experiment_dir", '../experiment', "Directory to store current experiment results")
FLAGS = flags.FLAGS


def main(_):
    pprint(flags.FLAGS.__flags)
    if not os.path.exists(FLAGS.experiment_dir):
        os.makedirs(FLAGS.experiment_dir)
        expt_num = "1"
    else:
        expts = os.listdir(FLAGS.experiment_dir)
        last_expr = max([int(folder) for folder in expts])
        expt_num = str(last_expr + 1)
    expt_result_path = os.path.join(FLAGS.experiment_dir, expt_num)
    os.makedirs(expt_result_path)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    chkpt_result_path = os.path.join(FLAGS.checkpoint_dir, expt_num)
    os.makedirs(chkpt_result_path)

    params_e_path = os.path.join(expt_result_path, "params.json")
    params_c_path = os.path.join(chkpt_result_path, "params.json")
    with open(params_e_path, 'w') as params_e, \
         open(params_c_path, 'w') as params_c:
        json.dump(flags.FLAGS.__flags, params_e)
        json.dump(flags.FLAGS.__flags, params_c)

    # Generate the indexes
    word2idx, field2idx, qword2idx, nF, max_words_in_table, word_set = \
        setup(FLAGS.data_dir, '../embeddings', FLAGS.n, FLAGS.batch_size,
              FLAGS.nW, FLAGS.min_field_freq, FLAGS.nQ)

    # Create the dataset objects
    train_dataset = DataSet(FLAGS.data_dir, 'train', FLAGS.n, FLAGS.nW, nF,
                            FLAGS.nQ, FLAGS.l, FLAGS.batch_size, word2idx,
                            field2idx, qword2idx,
                            FLAGS.max_fields, FLAGS.word_max_fields,
                            max_words_in_table, word_set)
    num_train_examples = train_dataset.num_examples()

    valid_dataset = DataSet(FLAGS.data_dir, 'valid', FLAGS.n, FLAGS.nW, nF,
                            FLAGS.nQ, FLAGS.l, FLAGS.batch_size, word2idx,
                            field2idx, qword2idx,
                            FLAGS.max_fields, FLAGS.word_max_fields,
                            max_words_in_table, word_set)

    test_dataset = DataSet(FLAGS.data_dir, 'test', FLAGS.n, FLAGS.nW, nF,
                           FLAGS.nQ, FLAGS.l, FLAGS.batch_size, word2idx,
                           field2idx, qword2idx,
                           FLAGS.max_fields, FLAGS.word_max_fields,
                           max_words_in_table, word_set)

    # The sizes of respective conditioning variables
    # for placeholder generation
    context_size = (FLAGS.n - 1)
    zp_size = context_size * FLAGS.word_max_fields
    zm_size = context_size * FLAGS.word_max_fields
    gf_size = FLAGS.max_fields
    gw_size = max_words_in_table
    copy_size = FLAGS.word_max_fields
    proj_size = FLAGS.nW + max_words_in_table

    # Generate the TensorFlow graph
    with tf.Graph().as_default():

        #Set the random seed for reproducibility
        tf.set_random_seed(1234)

        # Create the CopyAttention model
        model = CopyAttention(FLAGS.n, FLAGS.d, FLAGS.g, FLAGS.nhu,
                              FLAGS.nW, nF, FLAGS.nQ, FLAGS.l,
                              FLAGS.learning_rate, max_words_in_table,
                              FLAGS.max_fields, FLAGS.word_max_fields,
                              FLAGS.xavier)

        # Placeholders for train and validation
        context_pl, zp_pl, zm_pl, gf_pl, gw_pl, next_pl, copy_pl, proj_pl = \
            placeholder_inputs(FLAGS.batch_size, context_size, zp_size,
                               zm_size, gf_size, gw_size, copy_size,
                               proj_size)

        # Placeholders for test
        context_plt, zp_plt, zm_plt, gf_plt, gw_plt, copy_plt, proj_plt, next_plt = \
            placeholder_inputs_single(context_size, zp_size, zm_size,
                                      gf_size, gw_size, copy_size,
                                      proj_size)

        # Train and validation part of the model
        predict = model.inference(FLAGS.batch_size, context_pl, zp_pl, zm_pl,
                                  gf_pl, gw_pl, copy_pl, proj_pl)
        loss = model.loss(predict, next_pl)
        train_op = model.training(loss)
        # evaluate = model.evaluate(predict, next_pl)

        # Test component of the model
        # The batch_size parameter is replaced with 1.
        pred_single = model.inference(1, context_plt, zp_plt, zm_plt,
                                      gf_plt, gw_plt, copy_plt,
                                      proj_plt)
        predicted_label = model.predict(pred_single)

        # Initialize the variables and start the session
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(init)

        for epoch in range(1, FLAGS.num_epochs + 1):
            train_dataset.generate_permutation()
            start_e = time.time()
            for i in range(num_train_examples):
                feed_dict = fill_feed_dict(train_dataset, i,
                                           context_pl, zp_pl, zm_pl,
                                           gf_pl, gw_pl, next_pl,
                                           copy_pl, proj_pl)
                _, loss_value = sess.run([train_op, loss],
                                         feed_dict=feed_dict)

                if i % FLAGS.print_every == 0:
                    print "Epoch : %d\tStep : %d\tLoss : %0.3f" % (epoch, i, loss_value)

                if i == -1 and i % FLAGS.valid_every == 0:
                    print "Validation starting"
                    valid_loss = do_eval(sess, train_op, loss,
                                         valid_dataset,
                                         context_pl, zp_pl, zm_pl,
                                         gf_pl, gw_pl, next_pl,
                                         copy_pl, proj_pl)
                    print "Epoch : %d\tValidation loss: %0.5f" % (i, valid_loss)

                if i != 0 and i % FLAGS.sample_every == 0:
                    test_dataset.reset_context()
                    pos = 0
                    len_sent = 0
                    prev_predict = word2idx['<start>']
                    res_path = os.path.join(expt_result_path, 'sample.txt')
                    with open(res_path, 'a') as exp:
                        while pos != 1:
                            feed_dict_t, idx2wq = fill_feed_dict_single(test_dataset,
                                                                        prev_predict,
                                                                        0, context_plt,
                                                                        zp_plt, zm_plt,
                                                                        gf_plt, gw_plt,
                                                                        next_plt,
                                                                        copy_plt,
                                                                        proj_plt)
                            prev_predict = sess.run([predicted_label],
                                                    feed_dict=feed_dict_t)
                            prev = prev_predict[0][0][0]
                            if prev in idx2wq:
                                exp.write(idx2wq[prev] + ' ')
                                len_sent = len_sent + 1
                            else:
                                exp.write('<unk> ')
                                len_sent = len_sent + 1
                            if prev == word2idx['.']:
                                pos = 1
                                exp.write('\n')
                            if len_sent == 50:
                                break
                            prev_predict = prev

            duration_e = time.time() - start_e
            print "Time taken for epoch : %d is %0.3f minutes" % (epoch, duration_e/60)

            print "Saving checkpoint for epoch %d" % (epoch)
            checkpoint_file = os.path.join(chkpt_result_path, str(epoch) + '.ckpt')
            saver.save(sess, checkpoint_file)

            print "Validation starting"
            start = time.time()
            valid_loss = do_eval(sess, train_op, loss,
                                 valid_dataset, context_pl,
                                 zp_pl, zm_pl, gf_pl, gw_pl,
                                 next_pl, copy_pl, proj_pl)
            duration = time.time() - start
            print "Epoch : %d\tValidation loss: %0.5f" % (epoch, valid_loss)
            print "Time taken for validating epoch %d : %0.3f" % (epoch, duration)
            valid_res = os.path.join(expt_result_path, 'valid_loss.txt')

            with open(valid_res, 'a') as vloss_f:
                vloss_f.write("Epoch : %d\tValidation loss: %0.5f" % (epoch, valid_loss))


if __name__ == "__main__":
    tf.app.run()
