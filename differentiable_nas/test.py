def test():

  hyperparameter_w = tf.get_variable(
      "hyperparameter_w", [3, 4],
      dtype=tf.float32,
      initializer=tf.random_uniform_initializer)

  #logits = tf.matmul(input, output_w) + output_b

  zero_one_hyperparameter_w = hyperparameter_w[0]

  zero_one_hyperparameter_w_softmax = tf.nn.softmax(zero_one_hyperparameter_w)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    hyperparameter_w_value = sess.run(hyperparameter_w)
    print(hyperparameter_w_value)

    print(sess.run(zero_one_hyperparameter_w))
    print(sess.run(zero_one_hyperparameter_w_softmax))
