#!/usr/bin/env python3
import tensorflow as tf


class Yolo:
    """
    Yolo class that uses the Yolo v3 algorithm to perform object detection.
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Yolo class constructor.

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
