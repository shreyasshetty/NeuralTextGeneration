# pylint: disable=missing-docstring
import tensorflow as tf
import numpy as np

from input_data import get_hpca_embeddings


class CopyAttention(object):
    """ CopyAttention : The Neural Network model implemented
    as a class.
    """
    def __init__(self, n, d, g, nhu, nW, nF, nQ,
                 l, lr, max_words, max_fields,
                 word_max_fields, xavier=True):
        """ Initialize the CopyAttention model.

        Args:
            n                  : n-gram model parameter.
            d                  : word embedding dimension.
            g                  : global embedding dimension.
            nhu                : num. of hidden units.
            nW                 : Vocabulary size.
            nF                 : Field vocabulary size.
            nQ                 : Table vocabulary size.
            l                  : max. words per field.
            lr                 : learning rate for SGD.
            max_words          : Max. words in a table.
            max_fields         : Max. fields in a table.
            word_max_fields    : Max. num of fields a word
                                 can appear.
            xavier             : Initialize using Xavier
                                 initialization
        """
        print "Initializing the CopyAttention model"

        self._n = n
        self._d = d
        self._g = g
        self._nhu = nhu
        self._l = l
        self._lr = lr
        self._max_words = max_words
        self._max_fields = max_fields
        self._word_max_fields = word_max_fields

        # Input representation size.
        # i.e. concatenated input vector to the
        # model.
        d_1 = (n - 1)*3*d + 2*g

        with tf.name_scope('embeddings'):
            # Word embeddings
            embed = get_hpca_embeddings('../embeddings', d, nW)
            self._W = tf.to_float(tf.Variable(embed, trainable=True,
                                              name="word_embeddings"))

            # Local conditioning embeddings.
            # Local conditioning embedding flattened to a 2-D tensor
            # Contiguous set of 2l rows correpond to a field
            # First l rows correpond to embeddings from the start and
            # next l rows correpond to embeddings from the end
            # start(p) and end(n) embeddings for field i given by
            # positions
            # i*2*l + p, i*2*l + l + n respectively
            if xavier:
                # Local conditioning embeddings.
                self._Z_plus = tf.get_variable("zplus_embedding",
                                               shape=[l*nF, d],
                                               initializer=
                                               tf.contrib.layers.xavier_initializer())
                self._Z_minus = tf.get_variable("zminus_embedding",
                                                shape=[l*nF, d],
                                                initializer=
                                                tf.contrib.layers.xavier_initializer())

                # Global conditioning embeddings.
                # Global field conditioning.
                self._Gf = tf.get_variable("global_field_embedding",
                                           shape=[nF, g],
                                           initializer=
                                           tf.contrib.layers.xavier_initializer())

                # Global word conditioning.
                # NOTE: Here we differ from the paper
                #       Section 4.1: Table embeddings
                #       We define Gw to be a matrix of dimension (nQ,g)
                #       rather than nWxg to account for differences in
                #       vocabularies
                self._Gw = tf.get_variable("global_word_embedding",
                                           shape=[nQ, g],
                                           initializer=
                                           tf.contrib.layers.xavier_initializer())

                # Copy actions embedding.
                # Contiguous set of l embeddings correspond to a field
                # Field j position i indexed by j*l + i
                self._F_ji = tf.get_variable("copy_action_embedding",
                                             shape=[l*nF, d],
                                             initializer=
                                             tf.contrib.layers.xavier_initializer())

            else:
                # Local conditioning embeddings.
                w_range = np.sqrt(6)/(l*nF + d)
                self._Z_plus = tf.Variable(tf.random_uniform([l*nF, d],
                                                             minval=-w_range,
                                                             maxval=w_range),
                                           name="zplus_embedding",
                                           trainable=True)
                self._Z_minus = tf.Variable(tf.random_uniform([l*nF, d],
                                                              minval=-w_range,
                                                              maxval=w_range),
                                            name="zminus_embedding",
                                            trainable=True)

                # Global conditioning embeddings.
                # Global field conditioning.
                gf_range = np.sqrt(6)/(nF + g)
                self._Gf = tf.Variable(tf.random_uniform([nF, g],
                                                         minval=-gf_range,
                                                         max_val=gf_range),
                                       name="global_field_embedding",
                                       trainable=True)

                # Global word conditioning.
                # NOTE: Here we differ from the paper
                #       Section 4.1: Table embeddings
                #       We define Gw to be a matrix of dimension (nQ,g)
                #       rather than nWxg to account for differences in
                #       vocabularies
                gw_range = np.sqrt(6)/(nQ + g)
                self._Gw = tf.Variable(tf.random_uniform([nQ, g],
                                                         minval=-gw_range,
                                                         max_val=gw_range),
                                       name="global_word_embedding",
                                       trainable=True)

                # Copy actions embedding.
                # Contiguous set of l embeddings correspond to a field
                # Field j position i indexed by j*l + i
                fji_range = np.sqrt(6)/(l*nF + d)
                self._F_ji = tf.Variable(tf.random_uniform([l*nF, d],
                                                           minval=-fji_range,
                                                           maxval=fji_range),
                                         name="copy_action_embedding",
                                         trainable=True)

        with tf.name_scope("hidden_layer"):
            # Weights and biases
            if xavier:
                self._W_2 = tf.get_variable("hidden_layer_weights",
                                            shape=[d_1, nhu],
                                            initializer=
                                            tf.contrib.layers.xavier_initializer())
                self._b_2 = tf.Variable(tf.zeros([nhu]),
                                        name="input_layer_biases",
                                        trainable=True)
            else:
                w2_range = np.sqrt(6)/(d_1 + nhu)
                self._W_2 = tf.Variable(tf.random_uniform([d_1, nhu],
                                                          minval=-w2_range,
                                                          max_val=w2_range),
                                        name="hidden_layer_weights",
                                        trainable=True)
                self._b_2 = tf.Variable(tf.random_uniform([nhu],
                                                          minval=-w2_range,
                                                          maxval=w2_range),
                                        name="input_layer_biases",
                                        trainable=True)

        with tf.name_scope("output_layer"):
            # Weights and biases
            if xavier:
                self._W_3 = tf.get_variable("output_layer_weights",
                                            shape=[nhu, nW],
                                            initializer=
                                            tf.contrib.layers.xavier_initializer())
                self._b_3 = tf.Variable(tf.zeros([nW]),
                                        name="output_layer_biases",
                                        trainable=True)
            else:
                w3_range = np.sqrt(6)/(nW + nhu)
                self._W_3 = tf.Variable(tf.random_uniform([nhu, nW],
                                                          minval=w3_range,
                                                          maxval=-w3_range),
                                        name="output_layer_weights",
                                        trainable=True)
                self._b_3 = tf.Variable(tf.random_uniform([nW],
                                                          minval=-w3_range,
                                                          maxval=w3_range),
                                        name="output_layer_biases",
                                        trainable=True)

        with tf.name_scope("copy_action"):
            # Copy action weights and biases
            if xavier:
                self._W_4 = tf.get_variable("copy_action_weights",
                                            shape=[d, nhu],
                                            initializer=
                                            tf.contrib.layers.xavier_initializer())
                self._b_4 = tf.Variable(tf.zeros([nhu]),
                                        name="copy_action_biases",
                                        trainable=True)
            else:
                w4_range = np.sqrt(6)/(d + nhu)
                self._W_4 = tf.Variable(tf.random_uniform([d, nhu],
                                                          minval=-w4_range,
                                                          maxval=w4_range),
                                        name="copy_action_weights",
                                        trainable=True)
                self._b_4 = tf.Variable(tf.random_uniform([nhu],
                                                          minval=-w4_range,
                                                          maxval=w4_range),
                                        name="copy_action_biases",
                                        trainable=True)

        print "Done initializing the CopyAttention model"

    def _generate_x_ct(self, context, zp, zm, gf, gw, batch_size):
        """ generate_x_ct : Generate the input to the inference procedure.
        Implements equation (18) from https://arxiv.org/pdf/1603.07771v3.pdf
        Note that the z_ct input is split into zp and zm (plus and minus)
        embeddings.

        Args: context : Context inputs for the n-gram model.
                        (batch_size, (n-1))
              zp      : Local context, embeddings from the start of field.
                        Index into the Z_plus matrix.
                        (batch_size, word_max_fields*(n-1))
              zm      : Local context, embeddings from the end of field.
                        Index into the Z_minus matrix.
                        (batch_size, word_max_fields*(n-1))
              gf      : Global field context. (batch_size, max_fields*(n-1))
              gw      : Global word context. (batch_size, max_words*(n-1))

        Returns:
              psi_alpha_1(c_t, z_ct, g_f, g_w)      (18) from the paper
        """
        n = self._n
        d = self._d
        g = self._g
        word_max_fields = self._word_max_fields
        max_fields = self._max_fields
        max_words = self._max_words

        # Lookup the context embeddings.
        ct_lookup = tf.nn.embedding_lookup(self._W, context)
        # ct_lookup would return 1 d-dimensional embedding of each
        # context entry. We reshape the lookup result to obtain
        # a concatenated vector for each input example per row.
        # The eventual result should be (batch_size, (n-1)*d)
        # (n-1) words in context, each is a d-dimensional embedding.
        ct = tf.reshape(ct_lookup, (batch_size, (n-1)*d))

        # Lookup for local context embeddings.
        zp_lookup = tf.reshape(tf.nn.embedding_lookup(self._Z_plus, zp),
                               (batch_size, (n-1), word_max_fields, d))
        zp_fin = tf.reshape(tf.reduce_max(zp_lookup, reduction_indices=[2]),
                            (batch_size, (n-1)*d))

        zm_lookup = tf.reshape(tf.nn.embedding_lookup(self._Z_minus, zm),
                               (batch_size, (n-1), word_max_fields, d))
        zm_fin = tf.reshape(tf.reduce_max(zm_lookup, reduction_indices=[2]),
                            (batch_size, (n-1)*d))

        gf_lookup = tf.reshape(tf.nn.embedding_lookup(self._Gf, gf),
                               (batch_size, max_fields, g))
        gf_fin = tf.reshape(tf.reduce_max(gf_lookup, reduction_indices=[1]),
                            (batch_size, g))

        gw_lookup = tf.reshape(tf.nn.embedding_lookup(self._Gw, gw),
                               (batch_size, max_words, g))
        gw_fin = tf.reshape(tf.reduce_max(gw_lookup, reduction_indices=[1]),
                            (batch_size, g))

        # Concatenate all the embeddings to get the eventual
        # embedding lookup to feed the inference procedure.
        # (batch_size, d_1) sized lookup.
        # Concatenate horizontally.
        x_ct = tf.concat(1, (ct, zp_fin, zm_fin, gf_fin, gw_fin))
        return x_ct

    def _generate_copy(self, copy, h_ct, proj):
        """ Implement the copy action score for the
        CopyAttention model.

        Implements equation (21) and (22) from the paper
        https://arxiv.org/pdf/1603.07771v3.pdf
        """
        word_max_fields = self._word_max_fields
        d = self._d
        nhu = self._nhu
        num_words_in_table = tf.shape(copy)[0]
        copy_lookup = tf.reshape(tf.nn.embedding_lookup(self._F_ji, copy),
                                 (num_words_in_table*word_max_fields, d))
        q_wval = tf.reshape(tf.nn.xw_plus_b(copy_lookup, self._W_4, self._b_4),
                            (num_words_in_table, word_max_fields, nhu))
        q_w = tf.reduce_max(q_wval, reduction_indices=[1])
        copy_score = tf.matmul(tf.matmul(h_ct, tf.transpose(q_w)), proj)

        return copy_score

    def inference(self, batch_size, context, zp, zm, gf, gw,
                  copy, proj):
        """ Build the model to the point where it
        can be used for inference.

        Args:
            context  : Context word embedding indices for the current
                       batch. Each row corresponds to a context and
                       values represent indices of the words appearing
                       in the context. Size: [batch_size, (n-1)]
            zp       : Indices into the embedding matrix for local
                       context(plus).
                       Size: [batch_size, (n-1)*word_max_fields]
            zm       : Indices into the embedding matrix for local
                       context(minus).
                       Size: [batch_size, (n-1)*word_max_fields]
            gf       : Index into the global field embedding matrix.
                       Size: [batch_size, max_fields]
            gw       : Index into the global word embedding matrix.
                       Size: [batch_size, max_words]
            copy     : Index into the copy action embedding matrix.
                       Size: []
            proj     : Projection matrix to project out the copy
                       action score to output vocabulary.
                       Size: []

        Returns:
            The computed logits.
            Size: [batch_size, nW + max_words]
        """
        x_ct = self._generate_x_ct(context, zp, zm, gf, gw, batch_size)
        h_ct = tf.tanh(tf.nn.xw_plus_b(x_ct, self._W_2, self._b_2))
        phi_w = tf.nn.xw_plus_b(h_ct, self._W_3, self._b_3)

        pad = tf.zeros([batch_size, self._max_words])
        phi = tf.concat(1, (phi_w, pad))

        copy_score = self._generate_copy(copy, h_ct, proj)
        # Add up score from the feed-forward network
        # and the copy action score.
        logits = tf.add(copy_score, phi)

        return logits

    def loss(self, logits, next_word):
        """Calculate the cross entropy loss from logits

        Args:
            logits    : Logits tensor
                        [batch_size, nW + max_words]
            next_word : True next word tensor - [batch_size]

        Returns:
            loss : Loss tensor
        """
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits,
                                                                       next_word,
                                                                       name='xentropy')
        loss = tf.reduce_mean(cross_entropy, name='xentropy_sum')
        return loss

    def training(self, loss):
        """ Set up training

        Args:
            loss     : Tensor returned by loss

        Returns:
            train_op : Train operation for training
        """
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        train_op = optimizer.minimize(loss)
        return train_op

    def predict(self, logits):
        """ Predict the output based on logits.

        Args:
            logits : Logits tensor.
        """
        y = tf.nn.softmax(logits)
        predict = tf.argmax(y, 1)
        size = tf.shape(predict)[0]
        return tf.reshape(predict, (1, size))

    def evaluate(self, logits, next_word):
        correct = tf.nn.in_top_k(logits, next_word, 1)
        return tf.reduce_sum(tf.cast(correct, tf.int32))
