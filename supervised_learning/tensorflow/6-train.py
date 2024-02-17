#!/usr/bin/env python3
""" train function """
import tensorflow.compat.v1 as tf

create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_train_op = __import__('5-create_train_op').create_train_op


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    Builds, trains, and saves a neural network classifier.

    Args:
        X_train (ndarray): Matrix containing the training input data.
        Y_train (ndarray): Matrix containing the training labels.
        X_valid (ndarray): Matrix containing the validation input data.
        Y_valid (ndarray): Matrix containing the validation labels.
        layer_sizes (list): Contains the number of nodes in each layer of the
            network.
        activations (list): Contains the activation functions for each layer
            of the network.
        alpha (float): The learning rate.
        iterations (int): The number of iterations to train over.
        save_path (str, optional): Designates where to save the model.
            Defaults to "/tmp/model.ckpt".

    Returns:
        The path where the model was saved.
    """
    nx = X_train.shape[1]
    classes = Y_train.shape[1]

    x, y = create_placeholders(nx, classes)

    y_pred = forward_prop(x, layer_sizes, activations)

    loss = calculate_loss(y, y_pred)

    accuracy = calculate_accuracy(y, y_pred)

    train_op = create_train_op(loss, alpha)

    init_op = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)

        tf.add_to_collection("x", x)
        tf.add_to_collection("y", y)
        tf.add_to_collection("y_pred", y_pred)
        tf.add_to_collection("loss", loss)
        tf.add_to_collection("accuracy", accuracy)
        tf.add_to_collection("train_op", train_op)

        for i in range(iterations + 1):
            training_loss, training_acc = sess.run(
                [loss, accuracy], feed_dict={x: X_train, y: Y_train})

            valid_loss, valid_acc = sess.run(
                [loss, accuracy], feed_dict={x: X_valid, y: Y_valid})

            if i % 100 == 0 or i == iterations:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(training_loss))
                print("\tTraining Accuracy: {}".format(training_acc))
                print("\tValidation Cost: {}".format(valid_loss))
                print("\tValidation Accuracy: {}".format(valid_acc))

            if i < iterations:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})

        save_path = saver.save(sess, save_path)

    return save_path
