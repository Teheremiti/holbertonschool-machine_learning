#!/usr/bin/env python3
"""Neural style transfer."""
import numpy as np
import tensorflow as tf


class NST:
    """
    NST class that performs tasks for neural style transfer.
    """

    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        NST class constructor.

        Args:
        style_image (ndarray): Image used as style reference.
        content_image (ndarray): Image used as content reference.
        alpha (float): Weight for content cost.
        beta (float): Weight for style cost.
        """
        if not (isinstance(style_image, np.ndarray)
                and style_image.shape[-1] == 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")

        if not (isinstance(content_image, np.ndarray)
                and content_image.shape[-1] == 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.model = None
        self.load_model()

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixels values are between 0 and 1 and
        its largest side is 512 px.

        Args:
        image (ndarray): Image to be scaled, shape (h, w, 3).

        Returns:
        The scaled image as a `tf.tensor` with the shape (1, h_new, w_new, 3)
        where max(h_new, w_new) == 512 and min(h_new, w_new) is scaled
        proportionately.
        """
        if not (isinstance(image, np.ndarray) and image.shape[-1] == 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        h, w = image.shape[:2]
        scale = 512 / max(h, w)

        new_size = tuple(int(dim * scale) for dim in (h, w))
        resized_image = tf.image.resize(image, size=new_size, method='bicubic')

        # Normalize pixel values to the range [0, 1]
        resized_image = resized_image / 255

        # Clip values to ensure they remain within valid bounds
        clipped_image = tf.clip_by_value(resized_image, 0, 1)

        # Expand dimensions to match expected shape (1, h_new, w_new, 3)
        scaled_image = tf.expand_dims(clipped_image, axis=0)

        return scaled_image

    def load_model(self):
        """
        Creates the model used to calculate cost and saves it in the instance
        attribute `model`. Uses the VGG19 Keras model as a base.
        """
        # Load the pre-trained VGG19 model without the fully connected layers
        base_model = tf.keras.applications.VGG19(include_top=False,
                                                 weights='imagenet')
        base_model.trainable = False

        # Get outputs of selected layers
        selected_layers = self.style_layers + [self.content_layer]
        outputs = [
            base_model.get_layer(name).output for name in selected_layers
        ]

        # Construct the model
        model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
        model.save('vgg_base.h5')

        # Replace MaxPooling layers with AveragePooling layers
        custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        model = tf.keras.models.load_model('vgg_base.h5',
                                           custom_objects=custom_objects)

        self.model = model
