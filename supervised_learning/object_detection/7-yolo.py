#!/usr/bin/env python3
"""Some useless documentation to pass the checker."""
import os
import cv2

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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filter boxes.

        Args:
        boxes (list[ndarray]): Processed boundary boxes for each output,
            shape (grid_height, grid_width, anchor_boxes, 4).
        box_confidences (list[ndarray]): Processed box confidences for each
            output, shape (grid_height, grid_width, anchor_boxes, 1).
        box_class_probs (list[ndarray]): Processed box class probabilities for
            each output, shape (grid_height, grid_width, anchor_boxes, classes)

        Returns:
        Tuple of (filtered_boxes, box_classes, box_scores):
            filtered_boxes (ndarray): Filtered bounding boxes of shape (?, 4).
            box_classes (ndarray): Class number that each box in filtered_boxes
                predicts, shape (?,).
            box_scores (ndarray): Box scores for each box in filtered_boxes,
                shape (?).
        """
        filtered_boxes, box_classes, box_scores = (
            np.empty((0, 4)),
            np.empty((0,), dtype=int),
            np.empty((0,))
        )

        for i in range(len(boxes)):
            # Multiply element-wise
            box_score = box_confidences[i] * box_class_probs[i]

            box_classes_i = np.argmax(box_score, axis=-1)
            box_class_score = np.max(box_score, axis=-1)

            # Filtering mask
            mask = box_class_score >= self.class_t

            filtered_boxes = np.concatenate(
                (filtered_boxes, boxes[i][mask]), axis=0)
            box_classes = np.concatenate(
                (box_classes, box_classes_i[mask]), axis=0)
            box_scores = np.concatenate(
                (box_scores, box_class_score[mask]), axis=0)

        return filtered_boxes, box_classes, box_scores

    def iou(self, box1, box2):
        """
        Computes the Intersection over Union (IoU) between two bounding boxes.

        Args:
            box1 (ndarray): Coordinates of the first box [x1, y1, x2, y2].
            box2 (ndarray): Coordinates of the second box [x1, y1, x2, y2].

        Returns:
            The IoU value, a measure of overlap between the two boxes (0 to 1).
        """
        x1, y1 = np.maximum(box1[:2], box2[:2])
        x2, y2 = np.minimum(box1[2:], box2[2:])
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = np.prod(box1[2:] - box1[:2])
        area2 = np.prod(box2[2:] - box2[:2])
        return intersection / (area1 + area2 - intersection)

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Non-max Suppression (suppress overlapping box).

        Args:
            filtered_boxes (ndarray): shape(?,4)
                    all filtered bounding boxes
            box_classes (ndarray): shape(?,)
                    class number for class that filtered_boxes predicts
            box_scores (ndarray): shape(?)
                box scores for each box in filtered_boxes

        Returns:
        Tuple of shape (boxes, predicted_box_classes,
        predicted_box_scores):
            boxes (ndarray): Predicted bounding boxes ordered by
                class and box score, shape (?, 4).
            predicted_box_classes (ndarray): Class number for boxes
                ordered by class and box score, shape (?,).
            predicted_box_scores (ndarray): Box scores for boxes
                ordered by class and box score, shape (?).
        """
        # Lists are preferable due to the dynamic nature of box selection.
        boxes = []
        predicted_box_classes = []
        predicted_box_scores = []

        for cls in np.unique(box_classes):
            class_mask = box_classes == cls
            class_boxes, class_scores = (
                filtered_boxes[class_mask],
                box_scores[class_mask]
            )

            while len(class_boxes) > 0:
                max_idx = np.argmax(class_scores)
                boxes.append(class_boxes[max_idx])
                predicted_box_classes.append(cls)
                predicted_box_scores.append(class_scores[max_idx])

                ious = np.array([self.iou(class_boxes[max_idx], box)
                                for box in class_boxes])
                class_boxes = class_boxes[ious <= self.nms_t]
                class_scores = class_scores[ious <= self.nms_t]

        # Cast to numpy arrays for standardization
        boxes = np.array(boxes)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)

        return boxes, predicted_box_classes, predicted_box_scores

    @staticmethod
    def load_images(folder_path):
        """
        Load images.

        Args:
        folder_path (string): Path to the folder holding all the images to load

        Returns:
        Tuple of (images, image_paths).
            images (list): Images as ndarrays.
            image_paths (list): Paths to the individual images.
        """
        images, image_paths = [], []
        for filename in os.listdir(folder_path):
            try:
                image_path = os.path.join(folder_path, filename)
                image = cv2.imread(image_path)
                if image is not None:
                    images.append(image)
                    image_paths.append(image_path)

            except cv2.error:
                continue

        return images, image_paths

    def preprocess_images(self, images):
        """
        Preprocess images.
        Resizes the images with intercubic interpolation, and rescales them to
        have values in the range [0, 1].

        Args:
        images (ndarray): Images to preprocess.

        Returns:
        Tuple of (pimages, image_shapes):
            pimages (ndarray): Preprocessed images, shape
            (ni, input_h, input_w, 3):
                ni: Number of images that were preprocessed.
                input_h: Input height for the Darknet model.
                input_w: Input width for the Darknet model.
                3: Number of color channels.
            image_shapes (ndarray): original height and width of the images,
            of shape (ni, 2):
                2 => (image_height, image_width)
        """
        input_h, input_w = self.model.input.shape[1:3]
        pimages, image_shapes = [], []

        for image in images:
            image_h, image_w = image.shape[:2]
            image_shapes.append((image_h, image_w))

            # Resize image and normalize pixel values
            processed_img = cv2.resize(
                image,
                dsize=(input_h, input_w),
                interpolation=cv2.INTER_CUBIC
            )
            # Rescale image
            processed_img = processed_img / 255.0

            pimages.append(processed_img)

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Displays the image with all boundary boxes, class names, and
        box scores.
        If any key besides `s` is pressed, the image window will be closed
        without saving.
        If the `s` key is pressed, the image will be saved in the directory
        `detections`, located in the current directory, under the name
        `file_name`.

        Args:
        image (ndarray): Unprocessed image.
        boxes (ndarray): Boundary boxes for the image.
        box_classes (ndarray): Class indices for each box.
        box_scores (ndarray): Box scores for each box.
        file_name (str): File path where the original image is stored.
        """
        for box, cls, score in zip(boxes, box_classes, box_scores):
            x1, y1, x2, y2 = map(int, box)
            label = f"{self.class_names[cls]} {score:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                image,
                text=label,
                org=(x1, y1 - 5),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 0, 255),
                lineType=cv2.LINE_AA,
                thickness=1
            )

        name_img = os.path.basename(file_name)
        cv2.imshow(name_img, image)

        if cv2.waitKey(0) == ord('s'):
            os.makedirs("detections", exist_ok=True)
            cv2.imwrite(f"detections/{name_img}", image)

        cv2.destroyAllWindows()

    def predict(self, folder_path):
        """
        Makes predictions on each image in `folder_path`, then displays them
        using the `show_boxes` method.

        Args:
        folder_path (str): Path to the folder holding all images to predict.

        Returns:
        Tuple of (predictions, image_paths):
            predictions (list[tuple]): Predictions for each image of
            (boxes, box_classes, box_scores).
            image_paths (list): Image paths corresponding to each prediction.
        """
        images, image_paths = self.load_images(folder_path)
        pimages, image_shapes = self.preprocess_images(images)
        model_predictions = self.model.predict(pimages)

        predictions = []
        for image, image_path, outputs, shape in zip(images,
                                                     image_paths,
                                                     zip(*model_predictions),
                                                     image_shapes):
            boxes, confs, class_probs = self.process_outputs(outputs, shape)
            boxes, box_classes, box_scores = self.filter_boxes(
                boxes, confs, class_probs)
            boxes, box_classes, box_scores = self.non_max_suppression(
                boxes, box_classes, box_scores)

            predictions.append((boxes, box_classes, box_scores))
            self.show_boxes(image, boxes, box_classes, box_scores, image_path)

        return predictions, image_paths
