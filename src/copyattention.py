import tensorflow as tf
import numpy as np 

# TODO: Implement HPCA embedding
#from input_data import get_hpca_embeddings

class CopyAttention(object):
	"""CopyAttention model defined as an object.
	"""

	# pylint: disable=unused-argument
	def __init__(self, n, d, g, nhu, nW, nF, 
				 nQ, nQpr, l, lr, max_words, max_fields, word_max_fields,
				 batch_size):
		""" Initialize the CopyAttention model.

		Args:
			n		   :	n -gram parameter
			d		   :  dimension of word embeddings
			g		   :  dimension for global conditioning embedding
			nhu		 :  number of hidden units
			nW		  :  number of words in vocabulary for introduction
			nF		  :  number of fields present in the dataset
			nQ		  :  Table vocabulary size
			l		   :  Max. number of tokens in the field
			lr		  :  Learning rate for SGD
		"""
		self.n = n
		self.d = d
		self.g = g
		self.nhu = nhu
		self.nW = nW
		self.nF = nF
		self.nQ = nQ
                self.nQpr = nQpr
		self.l = l
		self.lr = lr
		self.max_words = max_words
		self.max_fields = max_fields
		self.word_max_fields = word_max_fields
		self.batch_size = batch_size

		print ("Initializing the copyattention model")
		# Input representation size	
		d_1 = (n-1)*3*d + 2*g

		# Embeddings
		with tf.name_scope('embeddings'):
			# TODO: change the initialization of word embeddings to use
			# HPCA embeddings
			# self.W = getHPCAEmbeddings()
			#if init_hpca_embed:
			#	self.W = tf.convert_to_tensor(get_hpca_embeddings())
			#else:
			#	self.W = tf.Variable(tf.random_normal([nW, d], stddev=0.01))
			# nW+2  - 1. For 'UNK' 2. For 'START' symbol
			#self.W = tf.Variable(tf.random_normal([nW+2, d])) #, stddev=0.01))
			wrange = np.sqrt(6)/(nW + 2  + d)
			self.W = tf.Variable(tf.random_uniform([nW+2, d], minval=-1*wrange, maxval=wrange), name='word_embedding', trainable=True) #, stddev=0.01))

            # qprime embeddings
			qpr_range = np.sqrt(6)/(nQpr + d)
			self.Qpr = tf.Variable(tf.random_uniform([nQpr, d], minval=-1*qpr_range, maxval= qpr_range), name='qpr_embedding', trainable=True)

            #qprime scoring function
			w_score_range = np.sqrt(6)/(nhu + d)
			self.W_score = tf.Variable(tf.random_uniform([nhu,d], minval = -1*w_score_range, maxval = w_score_range), name='wscoring', trainable=True)

			# Local conditioning embedding flattened to a 2-D tensor
			# Contiguous set of 2l rows correpond to a field
			# First l rows correpond to embeddings from the start and
			# next l rows correpond to embeddings from the end
			# start(p) and end(n) embeddings for field i given by
			# positions
			#			i*2*l + p, i*2*l + l + n respectively
			zrange = np.sqrt(6)/(l*(nF+1) + d)
			self.Z_plus = tf.Variable(tf.random_uniform([l*(nF+1), d], minval=-1*zrange, maxval=zrange), name='zplus_embedding', trainable=True) #, stddev=0.01))
			self.Z_minus = tf.Variable(tf.random_uniform([l*(nF+1), d], minval=-1*zrange, maxval=zrange), name='zminus_embedding', trainable=True) #, stddev=0.01))

			# Global conditioning embeddings matrices
			gfrange = np.sqrt(6)/(nF+1+g)
			self.Gf = tf.Variable(tf.random_uniform([(nF+1), g], minval=-1*gfrange, maxval=gfrange), name='global_field_embedding', trainable=True)
			# NOTE: Here we differ from the paper
			#	   Section 4.1: Table embeddings
			#	   We define Gw to be a matrix of dimension nQxg
			#	   rather than nWxg to account for differences in
			#	   vocabularies
			#self.Gw = tf.Variable(tf.random_normal([(nQ+1), g], stddev=0.01))
			gwrange = np.sqrt(6)/(nQ + 1 + g)
			self.Gw = tf.Variable(tf.random_uniform([(nQ+1), g], minval=-1*gwrange, maxval=gwrange), name='global_word_embedding', trainable=True)
			# Copy actions embedding
			# Contiguous set of l embeddings correspond to a field
			# Field j position i indexed by j*l + i
                        frange = np.sqrt(6)/(l*nF + d)
			#self.F_ji = tf.Variable(tf.random_normal([l*nF, d], stddev=0.01))
			self.F_ji = tf.Variable(tf.random_uniform([l*nF, d], minval=-1*frange, maxval=frange), name='copy_action_embedding', trainable=True)

		with tf.name_scope('hidden_context'):
			#Xavier initialization range
			hrange = np.sqrt(6)/(d_1 + nhu)
			# Weights and biases
			self.W_1 = tf.Variable(tf.random_uniform([d_1, nhu], minval=-1*hrange, maxval=hrange), name='input_weights', trainable=True)
			self.b_1 = tf.Variable(tf.random_uniform([nhu], minval=-1*hrange, maxval=hrange), name='input_biases', trainable=True)

			h_copy_range = np.sqrt(6)/(d + nhu)
            #Copy action weights and biases
			self.W_4 = tf.Variable(tf.random_uniform([d, nhu], minval=-1*h_copy_range, maxval=h_copy_range), name='copy_weights', trainable=True)
			self.b_4 = tf.Variable(tf.random_uniform([nhu], minval=-1*h_copy_range, maxval=h_copy_range), name='copy_biases', trainable=True)

		with tf.name_scope('output_phi_w'):
			#Xavier initialization range
			outrange = np.sqrt(6)/(nW + nhu)
			# Weights and biases
			self.W_out = tf.Variable(tf.random_uniform([nhu, nW], minval=-1*outrange, maxval=outrange), name='output_weights', trainable=True)
			self.b_out = tf.Variable(tf.random_uniform([nW], minval=-1*outrange, maxval=outrange), name='output_biases', trainable=True)

			#self.W_out = tf.Variable(tf.random_uniform([nW, nhu], minval = -1*outrange, maxval=outrange))
			#b_o = tf.Variable(tf.random_uniform([nW, 1], minval=-1*outrange, maxval=outrange))
			#self.b_out = tf.tile(b_o, tf.pack([1, batch_size]))
			#self.b_out = tf.Variable(tf.random_normal([nW, batch_size], stddev=0.01))

		print("Done initializing the copyattention model")

	def get_params(self):
		"""Helper function to returns model parameters

		Returns: A tuple (n, d, g, nhu, nW)
		"""
		return self.n, self.d, self.g, self.nhu, self.nW

	def generate_x_ct(self, context, zp, zm, gf, gw, batch_size):
		""" generate_x_ct : Generate the input to the inference procedure.
		Implements equation (18) from https://arxiv.org/pdf/1603.07771v3.pdf
		Note that the z_ct input is split into zp and zm (plus and minus)
		embeddings.

		Args: context : Context inputs for the n-gram model. (batch_size, (n-1))
			  zp      : Local context, embeddings from the start of field.
			  			Index into the Z_plus matrix. (batch_size, word_max_fields*(n-1))
			  zm      : Local context, embeddings from the end of field.
			  			Index into the Z_minus matrix. (batch_size, word_max_fields*(n-1))
			  gf      : Global field context. (batch_size, max_fields*(n-1))
			  gw      : Global word context. (batch_size, max_words*(n-1))
		
		Returns:
			  psi_alpha_1(c_t, z_ct, g_f, g_w)      (18) from the paper	
		"""
		n, d, g, nhu, nW = self.get_params()
		word_max_fields = self.word_max_fields
		max_fields = self.max_fields
		max_words = self.max_words
		
		# Lookup the context embeddings.	
		ct_lookup = tf.nn.embedding_lookup(self.W, context)
		# ct_lookup would return 1 d-dimensional embedding of each
		# context entry. We reshape the lookup result to obtain
		# a concatenated vector for each input example per row.
		# The eventual result should be (batch_size, (n-1)*d)
		# (n-1) words in context, each is a d-dimensional embedding.
		ct = tf.reshape(ct_lookup, (batch_size, (n-1)*d))

		# Lookup for local context embeddings.
		
		#zp_lookup = tf.nn.embedding_lookup(self.Z_plus, zp) #,(batch_size, (n-1)*d*word_max_fields))
		zp_lookup = tf.reshape(tf.nn.embedding_lookup(self.Z_plus, zp), (batch_size, (n-1), word_max_fields, d))
		zp_fin = tf.reshape(tf.reduce_max(zp_lookup, reduction_indices=[2]), (batch_size, (n-1)*d))

		zm_lookup = tf.reshape(tf.nn.embedding_lookup(self.Z_minus, zm), (batch_size, (n-1), word_max_fields, d))
		zm_fin = tf.reshape(tf.reduce_max(zm_lookup, reduction_indices=[2]), (batch_size, (n-1)*d))

		gf_lookup = tf.reshape(tf.nn.embedding_lookup(self.Gf, gf), (batch_size, max_fields, g))
		gf_fin = tf.reshape(tf.reduce_max(gf_lookup, reduction_indices=[1]), (batch_size, g))

		gw_lookup = tf.reshape(tf.nn.embedding_lookup(self.Gw, gw), (batch_size, max_words, g))
		gw_fin = tf.reshape(tf.reduce_max(gw_lookup, reduction_indices=[1]), (batch_size, g))

		# Concatenate all the embeddings to get the eventual embedding lookup to
		# feed the inference procedure.
		# (batch_size, d_1) sized lookup.
		# Concatenate horizontally.
		x_ct = tf.concat(1, (ct, zp_fin, zm_fin, gf_fin, gw_fin))
		return x_ct

	def inference(self, context, zp, zm, gf, gw, copy, projection, batch_size):
		""" Build the CopyAttention model.

		Args:
			context_word  : Context word embedding indices for the current
							batch. Each row corresponds to a context and
							values represent indices of the words appearing
							in the context. Size: [batch_size, (n-1)]
			zp            : Indices into the embedding matrix for local context(plus)
							Size: [batch_size, (n-1)*word_max_fields]
			zm            : Indices into the embedding matrix for local context(minus)
							Size: [batch_size, (n-1)*word_max_fields]
			gf            : Index into the global field embedding matrix.
							Size: [batch_size, max_fields]
			gw            : Index into the global word embedding matrix.
							Size: [batch_size, max_words]
		
		Returns:
			The computed logits. Size: [batch_size, nW]
		"""

		x_ct = self.generate_x_ct(context, zp, zm, gf, gw, batch_size) 

		h_ct = tf.tanh(tf.nn.xw_plus_b(x_ct, self.W_1, self.b_1))


		num_words = tf.shape(copy)[0]
		copy_lookup = tf.reshape(tf.nn.embedding_lookup(self.F_ji, copy), (num_words*word_max_fields,d))
		q_w_mat = tf.nn.xw_plus_b(copy_lookup, self.W_4, self.b_4)
		q_w = tf.reduce_max(tf.reshape(q_w_mat, (num_words, word_max_fields, d), reduction_indices=[1]))	
		copy_score = tf.matmul(projection, tf.matmul(q_w, h_ct))

		phi_out = tf.nn.xw_plus_b(h_ct, self.W_out, self.b_out)
		nw = tf.shape(phi_out)[1]
		total = tf.shape(copy_score)[0]
		pad = tf.zeros()

	
		logits = tf.nn.xw_plus_b(h_ct, self.W_out, self.b_out)
		predictions = tf.argmax(logits, 1)
		return logits

	def inference_v(self, context, zp, zm, gf, gw):
		""" Build the CopyAttention model.

		Args:
			context_word  : Context word embedding indices for the current
							batch. Each row corresponds to a context and
							values represent indices of the words appearing
							in the context. Size: [batch_size, (n-1)]
			zp            : Indices into the embedding matrix for local context(plus)
							Size: [batch_size, (n-1)*word_max_fields]
			zm            : Indices into the embedding matrix for local context(minus)
							Size: [batch_size, (n-1)*word_max_fields]
			gf            : Index into the global field embedding matrix.
							Size: [batch_size, max_fields]
			gw            : Index into the global word embedding matrix.
							Size: [batch_size, max_words]
		
		Returns:
			The computed logits. Size: [batch_size, nW]
		"""

		x_ct = self.generate_x_ct(context, zp, zm, gf, gw, 1) 

		h_ct = tf.tanh(tf.nn.xw_plus_b(x_ct, self.W_1, self.b_1))
		logits = tf.nn.xw_plus_b(h_ct, self.W_out, self.b_out)
		predictions = tf.argmax(logits, 1)
		return logits

	def loss(self, logits , next_word):
		"""Calculate the loss from logits obtained by adding both the
		   weights from context and copy actions

		Args:
			out	 : Logits tensor - [batch_size, num_words]
			sentence: Sentence tensor - [batch_size]

		Returns:
			loss : Loss tensor
		"""
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits,
																	   next_word,
																	   name='xentropy')
		loss = tf.reduce_mean(cross_entropy, name='xentropy_sum')
		return loss

	def train(self, loss):
		""" Set up training

		Args:
			loss	 : Tensor returned by loss

		Returns:
			train_op : Train operation for training
		"""
		#(loss_val, _) = loss
		#train_op = tf.train.AdamOptimizer(1e-4).minimize(loss_val)
		#gvs = optimizer.compute_gradients(loss)
		#capped_gvs = [(tf.clip_by_value(grad, -0.5, 0.5) , var) for grad, var in gvs]
		#train_op = optimizer.apply_gradients(capped_gvs)
		optimizer = tf.train.GradientDescentOptimizer(self.lr)
		train_op = optimizer.minimize(loss)
		return train_op

	def training(self, loss):
		""" Set up training

		Args:
			loss	 : Tensor returned by loss

		Returns:
			train_op : Train operation for training
		"""
		#(loss_val, _) = loss
		#train_op = tf.train.AdamOptimizer(1e-4).minimize(loss_val)
		#gvs = optimizer.compute_gradients(loss)
		#capped_gvs = [(tf.clip_by_value(grad, -0.5, 0.5) , var) for grad, var in gvs]
		#train_op = optimizer.apply_gradients(capped_gvs)
		optimizer = tf.train.GradientDescentOptimizer(self.lr)
		train_op = optimizer.minimize(loss)
		return train_op

	def predicted_label(self, logits):
		y = tf.nn.softmax(logits)
		predict = tf.argmax(y, 1)
		return predict

	def evaluate(self, logits, next_word):
		correct = tf.nn.in_top_k(logits, next_word, 1)
		return tf.reduce_sum(tf.cast(correct, tf.int32))
