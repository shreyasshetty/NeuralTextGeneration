# pylint: disable=missing-docstring
from __future__ import division

import time
import tensorflow as tf


def placeholder_inputs(batch_size, ct_size, zp_size, zm_size,
                       gf_size, gw_size, copy_size, proj_size):
    """ placeholder_inputs : Generate placeholders for train/valid.

    Args:
        batch_size : batch size for SGD.
        ct_size    : size of the word context input.
        zp_size    : z_plus conditioning size.
        zm_size    : z_minus conditioning size.
        gf_size    : global field conditioning size.
        gw_size    : global word conditioning size.
        copy_size  : size of the copy action inputs.
        proj_size  : projection matrix (copy action) size.

    Returns:
        Tensorflow placeholders for the inputs to the
        CopyAttention model.
    """
    context_pl = tf.placeholder(tf.int32, shape=(batch_size, ct_size))
    zp_pl = tf.placeholder(tf.int32, shape=(batch_size, zp_size))
    zm_pl = tf.placeholder(tf.int32, shape=(batch_size, zm_size))
    gf_pl = tf.placeholder(tf.int32, shape=(batch_size, gf_size))
    gw_pl = tf.placeholder(tf.int32, shape=(batch_size, gw_size))
    next_pl = tf.placeholder(tf.int32, shape=(batch_size))
    copy_pl = tf.placeholder(tf.int32, shape=(None, copy_size))
    proj_pl = tf.placeholder(tf.float32, shape=(None, proj_size))

    return context_pl, zp_pl, zm_pl, gf_pl, gw_pl, next_pl, copy_pl, proj_pl


def fill_feed_dict(data_set, pos, context_pl, zp_pl, zm_pl,
                   gf_pl, gw_pl, next_pl, copy_pl, proj_pl):
    """ fill_feed_dict : Obtain the next batch (train/valid)
    and place the data onto the placeholders.

    Args:
        data_set   : Dataset in consideration.
        pos        : Current index of the train/valid example.
        context_pl : Word context placeholder.
        zp_pl      : z_plus placeholder.
        zm_pl      : z_minus placeholder.
        gf_pl      : global field placeholder.
        gw_pl      : global word placeholder.
        next_pl    : next word placeholder.
        copy_pl    : copy action placeholder.
        proj_pl    : copy action projection matrix placeholder.

    Returns:
        feed_dict : Dictionary to feed the placeholders.
    """
    context, zp, zm, gf, gw, next_w, copy, proj = data_set.next_batch(pos)

    feed_dict = {
        context_pl: context,
        zp_pl: zp,
        zm_pl: zm,
        gf_pl: gf,
        gw_pl: gw,
        next_pl: next_w,
        copy_pl: copy,
        proj_pl: proj
    }

    return feed_dict


def placeholder_inputs_single(ct_size, zp_size, zm_size, gf_size,
                              gw_size, copy_size, proj_size):
    """ placeholder_inputs_single : Generate placeholders
    for test. Used for sequential sentence generation.

    Args:
        ct_size    : size of the word context input.
        zp_size    : z_plus conditioning size.
        zm_size    : z_minus conditioning size.
        gf_size    : global field conditioning size.
        gw_size    : global word conditioning size.
        copy_size  : size of the copy action inputs.
        proj_size  : projection matrix (copy action) size.

    Returns:
        Tensorflow placeholders for the inputs to the
        CopyAttention model.
    """
    context_pl = tf.placeholder(tf.int32, shape=(ct_size))
    zp_pl = tf.placeholder(tf.int32, shape=(zp_size))
    zm_pl = tf.placeholder(tf.int32, shape=(zm_size))
    gf_pl = tf.placeholder(tf.int32, shape=(gf_size))
    gw_pl = tf.placeholder(tf.int32, shape=(gw_size))
    copy_pl = tf.placeholder(tf.int32, shape=(None, copy_size))
    proj_pl = tf.placeholder(tf.float32, shape=(None, proj_size))
    next_pl = tf.placeholder(tf.int32, shape=(1))

    return context_pl, zp_pl, zm_pl, gf_pl, gw_pl, copy_pl, proj_pl, next_pl


def fill_feed_dict_single(data_set, prev, pos, context_pl, zp_pl, zm_pl,
                          gf_pl, gw_pl, next_pl, copy_pl, proj_pl):
    """ fill_feed_dict_single : Obtain the next sequential context and
    feed into the placeholders.

    Args:
        data_set   : Dataset in consideration. (Can be test/valid).
        prev       : Prediction at the previous step (index number).
        pos        : Index to the infobox for sentence generation.
        context_pl : Word context placeholder.
        zp_pl      : z_plus conditioning placeholder.
        zm_pl      : z_minus conditioning placeholder.
        gf_pl      : global field conditioning placeholder.
        gw_pl      : global word conditioning placeholder.
        next_pl    : next_word placeholder. (Not used. Retained
                     for consistency of interfaces.)
        copy_pl    : copy action input placeholder.
        proj_pl    : projection matrix (copy scores) placeholder.

    Returns:
        feed_dict : Dictionary to feed the placeholders.
        idx2wq    : Reverse index to the resized vocabulary.
                    Used for decoding sentences.
    """
    single_input = data_set.next_single(pos, prev)
    (context, zp, zm, gf, gw, _, copy, proj, idx2wq) = single_input

    feed_dict = {
        context_pl: context,
        zp_pl: zp,
        zm_pl: zm,
        gf_pl: gf,
        gw_pl: gw,
        copy_pl: copy,
        proj_pl: proj,
    }

    return feed_dict, idx2wq


def do_eval(sess, train_op, loss, data_set, context_pl, zp_pl,
            zm_pl, gf_pl, gw_pl, next_pl, copy_pl, proj_pl):
    """ do_eval : Perform cross entropy loss evaluation on the
    dataset.

    Args:
        sess       : The TensorFlow sess in which we desire to
                     evaluate the loss.
        train_op   : train tensor from within the CopyAttention model.
                     (part of the inference, train, loss setup.)
        loss       : loss tensor from the CopyAttention model.
        data_set   : Dataset to perform evaluation on.
        context_pl : word context placeholder.
        zp_pl      : z_plus conditioning placeholder.
        zm_pl      : z_minus conditioning placeholder.
        gf_pl      : global field conditioning placeholder.
        gw_pl      : global word conditioning placeholder.
        next_pl    : next word placeholder.
        copy_pl    : copy action placeholder.
        proj_pl    : copy score projection matrix placeholder.

    Returns:
        data_set_loss : Cross entropy loss on data_set.
    """
    start = time.time()
    total_loss = 0
    num_examples = data_set.num_examples()
    data_set.generate_permutation()

    # Compute the loss across the entire dataset.
    for i in range(num_examples):
        feed_dict = fill_feed_dict(data_set, i, context_pl, zp_pl, zm_pl,
                                   gf_pl, gw_pl, next_pl, copy_pl, proj_pl)
        _, loss_val = sess.run([train_op, loss], feed_dict=feed_dict)
        total_loss += loss_val

    # Loss on the dataset divided by the number of examples.
    data_set_loss = total_loss/num_examples
    duration = time.time() - start
    print "Time taken for validation : %0.3f s" % (duration)
    return data_set_loss
