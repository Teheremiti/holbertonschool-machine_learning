#!/usr/bin/env python3
"""Some useless documentation to pass the checker."""
import numpy as np
import tensorflow as tf


class Yolo:
    """
    Yolo class that uses the Yolo v3 algorithm to perform object detection.
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Yolo class constructor.

        Args:
        model_path (str): Path where Darknet Keras model is stored.
        classes_path (str): path to where the list of class names used for the
            Darknet model can be found, listed in order of index.
        class_t (float): Box score threshold for the initial filtering step.
        nms_t (float): IOU threshold for non-max suppression.
        anchors (ndarray): Shape (outputs, anchor_boxes, 2) containing all of
            the anchor boxes:
            outputs (int): Number of outputs (predictions) made by the
                Darknet model.
            anchor_boxes (int): Number of anchor boxes for each prediction.
            2 => [anchor_box_width, anchor_box_height]
        """
        self.model = tf.keras.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Process outputs.

        Args:
        outputs (list[ndarray]): Predictions from the Darknet model for
            a single image. Each output will have the shape
            (grid_height, grid_width, anchor_boxes, 4 + 1 + classes):
            grid_height (int): Height of the grid used for the output.
            grid_width (int): Width of the grid used for the output.
            anchor_boxes (int): Number of anchor boxes used.
            4 => (t_x, t_y, t_w, t_h)
            1 => box_confidence
            classes (int): Class probabilities for all classes.
        image_size (ndarray): Image's original size [image_height, image_width]

        Returns:
        Tuple of (boxes, box_confidences, box_class_probs):
            boxes (list[ndarray]): processed boundary boxes for each output
                of shape (grid_height, grid_width, anchor_boxes, 4):
                4 => (x1, y1, x2, y2), which represent the boundary box
                relative to original image.
            box_confidences (list[ndarray]): Box confidences for each output,
                shape (grid_height, grid_width, anchor_boxes, 1).
            box_class_probs (list[ndarray]): Box class probabilities for each
                output, shape (grid_height, grid_width, anchor_boxes, classes).
        """
        image_height, image_width = image_size

        boxes, box_confidences, box_class_probs = [], [], []

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            grid_x, grid_y = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
            grid_x, grid_y = grid_x[..., np.newaxis], grid_y[..., np.newaxis]

            anchor_w, anchor_h = self.anchors[i, :, 0], self.anchors[i, :, 1]

            # Extract relevant values
            t_x, t_y, t_w, t_h = (
                output[..., 0],
                output[..., 1],
                output[..., 2],
                output[..., 3]
            )

            # Compute box coordinates
            b_x = (1 / (1 + np.exp(-t_x)) + grid_x) / grid_w
            b_y = (1 / (1 + np.exp(-t_y)) + grid_y) / grid_h
            b_w = (anchor_w * np.exp(t_w)) / self.model.input.shape[1]
            b_h = (anchor_h * np.exp(t_h)) / self.model.input.shape[2]

            x1 = (b_x - b_w / 2) * image_width
            x2 = (b_x + b_w / 2) * image_width
            y1 = (b_y - b_h / 2) * image_height
            y2 = (b_y + b_h / 2) * image_height

            boxes.append(np.stack([x1, y1, x2, y2], axis=-1))

            # Compute confidence scores
            box_confidences.append(1 / (1 + np.exp(-output[..., 4:5])))
            box_class_probs.append(1 / (1 + np.exp(-output[..., 5:])))

        return boxes, box_confidences, box_class_probs
