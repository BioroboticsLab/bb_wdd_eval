import cv2
import numpy as np


class CoordinateTransformer:
    """
    Given the marker WDD and HD marker coordinates, finds a homography so
    coordinates can be transformed from one system to the other.
    """

    def __init__(self, src_markers, dst_markers):
        src_markers = np.array(src_markers, dtype=np.float32)
        dst_markers = np.array(dst_markers, dtype=np.float32)
        self.H, _ = cv2.findHomography(src_markers, dst_markers)

    def transform_coordinates(self, x, y):
        src_coordinates = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)
        dest_coordinates = cv2.perspectiveTransform(src_coordinates, self.H)
        x_transformed, y_transformed = dest_coordinates[0][0]
        return x_transformed, y_transformed
