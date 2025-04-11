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
        self.gram_style_features, self.content_feature = (
            self.generate_features())

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

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculates the Gram matrix.

        Args:
        input_layer (tf.Tensor | tf.Variable): Layer output whose Gram matrix
        should be calculated., of shape (1, h, w, c).

        Returns:
        The tf.Tensor of shape (1, c, c) containing the Gram matrix.
        """
        if (not isinstance(input_layer, (tf.Tensor, tf.Variable))
                or input_layer.shape.rank != 4):
            raise TypeError("input_layer must be a tensor of rank 4")

        batch_size, height, width, channels = tf.unstack(tf.shape(input_layer))

        # Reshape to (batch_size, height*width, channels)
        new_shape = [batch_size, height * width, channels]
        reshaped = tf.reshape(input_layer, new_shape)

        # Compute Gram matrix as the inner product of the reshaped input layer
        gram = tf.matmul(reshaped, reshaped, transpose_a=True)

        # Normalize by the number of spatial locations (height * width)
        num_positions = tf.cast(height * width, tf.float32)
        gram_normalized = gram / num_positions

        return gram_normalized

    def generate_features(self):
        """
        Extracts the style and content features.
        Sets the public instance attributes:
            gram_style_features (tf.Tensor[]): Gram matrices calculated from
            the style layer outputs of the style image.
            content_feature: Content layer output of the content image.

        Returns:
        The public instance attributes `gram_style_features`
        and `content_feature`.
        """
        # Preprocess input images
        style_image_p = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255)
        content_image_p = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255)

        # Extract intermediate layer outputs
        style_layer_outputs = self.model(style_image_p)
        content_layer_outputs = self.model(content_image_p)

        # Compute Gram matrices for style representation (excluding last layer)
        self.gram_style_features = [
            self.gram_matrix(output) for output in style_layer_outputs[:-1]
        ]

        # Use the last layer for content representation
        self.content_feature = content_layer_outputs[-1]

        return self.gram_style_features, self.content_feature

    def layer_style_cost(self, style_output, gram_target):
        """
        Calculates the style cost for a single layer.

        Args:
        style_output (tf.Tensor): Layer style output of the generated image, of
        shape (1, h, w, c).
        gram_target (tf.Tensor): Gram matrix of the target style output for
        that layer, of shape (1, c, c).

        Returns:
        tf.Tensor: Scalar representing the style cost for this layer.
        """
        if (not isinstance(style_output, (tf.Tensor, tf.Variable))
                or style_output.shape.rank != 4):
            raise TypeError("style_output must be a tensor of rank 4")

        channels = style_output.shape[-1]

        expected_shape = [1, channels, channels]
        if (not isinstance(gram_target, (tf.Tensor, tf.Variable))
                or gram_target.shape != expected_shape):
            raise TypeError(
                f"gram_target must be a tensor of shape {expected_shape}")

        gram_matrix = self.gram_matrix(style_output)
        layer_style_cost = tf.reduce_mean(tf.square(gram_matrix - gram_target))

        return layer_style_cost

    def style_cost(self, style_outputs):
        """
        Calculates the total style cost.

        Args:
        style_outputs (tf.Tensor[]): Style outputs for the generated image.
        Each layer should be weighted evenly with all weights summing to 1.

        Returns:
        tf.Tensor: Scalar representing the total style cost.
        """
        expected_length = len(self.style_layers)
        if (not isinstance(style_outputs, list)
                or len(style_outputs) != expected_length):
            raise TypeError(
                f"style_outputs must be a list with a length of "
                f"{expected_length}"
            )

        layer_weight = 1.0 / expected_length

        style_cost = sum([
            layer_weight * self.layer_style_cost(style, target)
            for style, target in zip(style_outputs, self.gram_style_features)
        ])

        return style_cost

    def content_cost(self, content_output):
        """
        Calculates the content cost.

        Args:
        content_output (tf.Tensor): Content output for the generated image.

        Returns:
        tf.Tensor: Scalar representing the content cost.
        """
        expected_shape = self.content_feature.shape
        if (not isinstance(content_output, (tf.Tensor, tf.Variable)) or
                content_output.shape != expected_shape):
            raise TypeError(
                f"content_output must be a tensor of shape "
                f"{expected_shape}"
            )

        content_cost = tf.reduce_mean(
            tf.square(content_output - self.content_feature))

        return content_cost

    def total_cost(self, generated_image):
        """
        Calculates the total cost.

        Args:
        generated_image (tf.Tensor): Generated image of shape (1, nh, nw, 3).

        Returns:
        The tuple (J, J_content, J_style), where:
            J: Total cost.
            J_content: Content cost.
            J_style: Style cost.
        """
        expected_shape = self.content_image.shape
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable))
                or generated_image.shape != expected_shape):
            raise TypeError(
                f"generated_image must be a tensor of shape "
                f"{expected_shape}"
            )

        # Preprocess generated image
        preprocessed_image = tf.keras.applications.vgg19.preprocess_input(
            generated_image * 255)

        # Extract features from the generated image
        generated_outputs = self.model(preprocessed_image)
        generated_style_outputs = generated_outputs[:-1]
        generated_content_output = generated_outputs[-1]

        J_content = self.content_cost(generated_content_output)
        J_style = self.style_cost(generated_style_outputs)
        J = self.alpha * J_content + self.beta * J_style

        return J, J_content, J_style

    def compute_grads(self, generated_image):
        """
        Computes the gradients for the generated image.

        Args:
        generated_image (tf.Tensor | tf.Variable): Generated image of shape
        (1, nh, nw, 3).

        Returns:
        The tuple (gradients, J_total, J, content, J_style), where:
            gradients (tf.Tensor): Gradients for the generated image.
            J_total: Total cost for the generated image.
            J_content: Content cost for the generated image.
            J_style: Style cost for the generated image.
        """
        expected_shape = self.content_image.shape
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable))
                or generated_image.shape != expected_shape):
            raise TypeError(
                f"generated_image must be a tensor of shape "
                f"{expected_shape}"
            )

        with tf.GradientTape() as tape:
            tape.watch(generated_image)
            J_total, J_content, J_style = self.total_cost(generated_image)

        gradients = tape.gradient(J_total, generated_image)

        return gradients, J_total, J_content, J_style

    def generate_image(
            self, iterations=1000, step=None, lr=0.01, beta1=0.9, beta2=0.99):
        """
        Generates the neural style transfered image using Adam optimization.

        Args:
        iterations (int): Iterations to perform gradient descent over.
        step (int | None): The step at which information should be printed:
        lr (float): Learning rate for gradient descent.
        beta1 (float): The beta1 parameter for gradient descent.
        beta2 (float): The beta2 parameter for gradient descent.

        Returns:
        The best generated_image and best cost.
        """
        if not isinstance(iterations, int):
            exit(1)
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            print("ValueError iterations")
            exit(2)
            raise ValueError("iterations must be positive")

        if step is not None and not isinstance(step, int):
            print("TypeError step")
            exit(3)
            raise TypeError("step must be an integer")
        if not 0 < step <= iterations:
            print("ValueError step")
            exit(4)
            raise ValueError("step must be positive and less than iterations")

        if not isinstance(lr, (float, int)):
            exit(5)
            print("TypeError lr")
            raise TypeError("lr must be a number")
        if lr < 0:
            exit(6)
            print("ValueError lr")
            raise ValueError("lr must be positive")

        if not isinstance(beta1, float):
            print("TypeError beta1")
            exit(7)
            raise TypeError("beta1 must be a float")
        if not 0 <= beta1 <= 1:
            print("ValueError beta1")
            exit(8)
            raise ValueError("beta1 must be in the range [0, 1]")

        if not isinstance(beta2, float):
            print("TypeError beta2")
            exit(9)
            raise TypeError("beta2 must be a float")
        if not 0 <= beta2 <= 1:
            print("ValueError beta2")
            exit(10)
            raise ValueError("beta2 must be in the range [0, 1]")

        generated_image = tf.Variable(self.content_image)
        best_cost = float('inf')
        best_image = None

        optimizer = tf.optimizers.Adam(
            learning_rate=lr, beta_1=beta1, beta_2=beta2)

        for i in range(iterations + 1):
            grads, J_total, J_content, J_style = self.compute_grads(
                generated_image)
            optimizer.apply_gradients([(grads, generated_image)])

            if J_total < best_cost:
                best_cost = float(J_total)
                best_image = generated_image

            if step is not None and (i % step == 0 or i == iterations):
                print(
                    f"Cost at iteration {i}: "
                    f"{J_total}, content {J_content}, style {J_style}"
                )

        best_image = tf.clip_by_value(best_image[0], 0, 1).numpy()

        return best_image, best_cost
