import tensorflow as tf

def placeholder_inputs(batch_size, ct_size, zp_size, zm_size, gf_size, gw_size, copy_size, projection_size):
	context_pl = tf.placeholder(tf.int32, shape=(batch_size, ct_size))
	zp_pl = tf.placeholder(tf.int32, shape=(batch_size, zp_size))
	zm_pl = tf.placeholder(tf.int32, shape=(batch_size, zm_size))
	gf_pl = tf.placeholder(tf.int32, shape=(batch_size, gf_size))
	gw_pl = tf.placeholder(tf.int32, shape=(batch_size,gw_size))
	next_pl = tf.placeholder(tf.int32, shape=(batch_size))
	projection_pl = tf.placeholder(tf.int32, shape=(projection_size, None))
	tword_pl = tf.placeholder(tf.int32, shape=(None, copy_size))

	return context_pl, zp_pl, zm_pl, gf_pl, gw_pl, next_pl, copy_pl, projection_pl

def fill_feed_dict(data_set, pos, context_pl, zp_pl, zm_pl, gf_pl, gw_pl, next_pl, copy_pl, projection_pl):
	context, local_plus, local_minus, global_field, global_word, next_word, copy, projection_mat = data_set.next_batch(pos)
	
	feed_dict = {
		context_pl : context,
		zp_pl : local_plus,
		zm_pl : local_minus,
		gf_pl : global_field,
		gw_pl : global_word,
		next_pl : next_word,
		copy_pl : copy, 
		projection_pl : projection_mat
	}

	return feed_dict

def placeholder_inputs_single(ct_size, zp_size, zm_size, gf_size, gw_size):
	context_pl = tf.placeholder(tf.int32, shape=(ct_size))
	zp_pl = tf.placeholder(tf.int32, shape=(zp_size))
	zm_pl = tf.placeholder(tf.int32, shape=(zm_size))
	gf_pl = tf.placeholder(tf.int32, shape=(gf_size))
	gw_pl = tf.placeholder(tf.int32, shape=(gw_size))
	next_pl = tf.placeholder(tf.int32, shape=(1))

	return context_pl, zp_pl, zm_pl, gf_pl, gw_pl, next_pl

def fill_feed_dict_single(data_set, prev_predict, context_pl, zp_pl, zm_pl, gf_pl, gw_pl, next_pl):
	context, local_plus, local_minus, global_field, global_word, next_word = data_set.next_single(prev_predict)
	
	feed_dict = {
		context_pl : context,
		zp_pl : local_plus,
		zm_pl : local_minus,
		gf_pl : global_field,
		gw_pl : global_word,
	}

	return feed_dict
