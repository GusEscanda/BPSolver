from itertools import product
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from collections import Counter
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

class Color:
    """
        BGR colors
    """
    RED      = (0, 0, 255)
    GREEN    = (0, 255, 0)
    BLUE     = (255, 0, 0)
    YELLOW   = (0, 255, 255)
    ORGANGE  = (0, 128, 255)
    CYAN     = (255, 255, 0)
    MAGENTA  = (255, 0, 255)
    WHITE    = (255, 255, 255)
    BLACK    = (0, 0, 0)
    GRAY     = (128, 128, 128)

class Grid:

    OUTPUT_LINE_THICKNESS = 4
    OUTPUT_POINT_RADIUS = 10

    def __init__(self, image_path=None, image=None):
        """
        Analyzes an image to detect a grid (a set of horizontal and vertical lines at regular intervals).

        :param image_path: File name where the image to be analyzed is located, or None if the image is provided directly.
        :param image: ndarray containing the image to be analyzed, or None if the image is loaded from a file.
        """
        if image_path:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        self.image_path = image_path
        self.image = image
        self.height, self.width = self.image.shape[:2] if self.image is not None else (0, 0)
        self.work_imgs = {}
        self.all_grids = []
        self.n = 0
        self.x_axis, self.y_axis = [], []
        self.x_size, self.y_size = 0.0, 0.0

    def preprocess_image(
            self, 
            resize_width=None,
            resize_height=None,
            resize_factor=None,
            threshold_values=[15, 30, 50, 70, 100, 130, 160, 190, 210, 230, 240], 
            blur_ksize=(5, 5)
        ):
        """
        Image preprocessing: creates a set of images with different types of processing applied to the original image,
        which will later be used to detect the grid.
        The preprocessing includes:

            1) Resizing, in width and/or height.
            2) Converting to grayscale.
            3) Binarization, using different threshold values.
            4) Applying Gaussian blur.

        A dictionary is generated containing processed versions of the original image, combining these preprocessing steps.

        :param resize_width: New width. If not specified, the width remains unchanged, unless resize_height or resize_factor 
                             is specified, in which case the width is adjusted to maintain the aspect ratio.
        :param resize_height: New height. If not specified, the height remains unchanged, unless resize_width or resize_factor 
                              is specified, in which case the height is adjusted to maintain the aspect ratio.
        :param resize_factor: Scaling factor for both dimensions. If not specified, no resizing is performed,
                              unless resize_width and/or resize_height are provided, in which case they take precedence.

        :param threshold_values: List of threshold values for binarization, useful for processing images that may vary in brightness.
        :param blur_ksize: Parameter for GaussianBlur. Sometimes applying a slight blur can help detect lines more effectively.
        """
        # Handle resize parameters and set defaults
        width, height = self.width, self.height
        if resize_width or resize_height or resize_factor:
            # Calculate new width
            if resize_width:
                width = resize_width
            elif resize_factor:
                width = int((self.width * resize_factor) + 0.5)
            elif resize_height:
                width = int((self.width * resize_height / self.height) + 0.5)
            # Calculate new height
            if resize_height:
                height = resize_height
            elif resize_factor:
                height = int((self.height * resize_factor) + 0.5)
            elif resize_width:
                height = int((self.height * resize_width / self.width) + 0.5)

        # Resize if necessary
        image = self.image.copy()
        if width != self.width or height != self.height:
            method_w = cv2.INTER_CUBIC if width > self.width else cv2.INTER_AREA
            method_h = cv2.INTER_CUBIC if height > self.height else cv2.INTER_AREA
            method = method_w if method_h == method_w else cv2.INTER_LINEAR
            image = cv2.resize(image, (width, height), interpolation=method)

        # Build the work images
        if len(image.shape) == 2:
            img_gray = image.copy()
        elif len(image.shape) == 3 and image.shape[2] == 3:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise Exception('Invalid image format')

        self.work_imgs = {}
        self.work_imgs['GRAY'] = img_gray
        self.work_imgs['EQ-GRAY'] = cv2.equalizeHist(img_gray)

        binaries = {}
        for pr in self.work_imgs:
            for thr in threshold_values:
                _, binary = cv2.threshold(self.work_imgs[pr], thr, 255, cv2.THRESH_BINARY_INV)
                binaries[f'{pr}|BIN:{("000"+str(thr))[-3:]}'] = binary
        self.work_imgs = {**self.work_imgs, **binaries}

        blurs = {}
        for pr in self.work_imgs:
            blur = cv2.GaussianBlur(self.work_imgs[pr], blur_ksize, 0)
            blurs[f"{pr}|BLUR"] = blur
        self.work_imgs = {**self.work_imgs, **blurs}

    def find_grid(
            self,
            min_line_length=None,
            max_line_gap=None,
            max_groupping_dist=None,
            min_valid_n=3,
            max_valid_n=16,
            show_process=False
        ):
        """
        Finds grids in all preprocessed images using all possible combinations of the provided parameters,
        generates a list of all detected grids, and selects one as the 'best' grid, i.e., the one that appears
        most frequently and covers the largest area.

        Grids are represented as x_axis and y_axis vectors containing the x and y values of all horizontal and 
        vertical lines that compose the grid.

        The parameters min_line_length, max_line_gap, and max_groupping_dist are lists. Within each preprocessed image, 
        grids will be searched using all combinations of these parameter values. Both the selected grid and the 
        grids found in each preprocessed image remain stored in the object for potential analysis.

        :param min_line_length: List of possible values for the minimum line length (in pixels) required for a line 
                                to be considered during the process. If not specified, it defaults to the image width 
                                multiplied by 0.3, 0.4, 0.5, 0.6, 0.7, and 0.8.
        :param max_line_gap: List of possible values for the maximum gap (in pixels) allowed in the continuity of a line.
                            If not specified, it defaults to [8, 12, 16, 20].
        :param max_groupping_dist: List of possible values for the maximum distance (in pixels) within which two points 
                                or lines are considered the same. If not specified, it defaults to [3, 5, 8].
        :param min_valid_n: Minimum number of points per axis; default is 3.
        :param max_valid_n: Maximum number of points per axis; default is 16.
        :param show_process: Indicates whether to return a list of images showing partial processing results 
                            (detected lines and associated grids).
        :return: A list of processed images if show_process is True; otherwise, None.
        """

        if not self.work_imgs:
            self.preprocess_image()
        
        output = {} if show_process else None

        # Find valid grids in all preprocessed images
        self.all_grids = []
        for pr in self.work_imgs:

            # Set the parameters for grid detection in this image
            height, width = self.work_imgs[pr].shape[:2]
            if isinstance(min_line_length, (list, np.ndarray)):
                min_line_length_list = min_line_length
            else:
                min_line_length_list = [min_line_length] if min_line_length is not None else [width * i // 10 for i in range(3, 9)]
            if isinstance(max_line_gap, (list, np.ndarray)):
                max_line_gap_list = max_line_gap
            else:
                max_line_gap_list = [max_line_gap] if max_line_gap is not None else [8, 12, 16, 20]
            if isinstance(max_groupping_dist, (list, np.ndarray)):
                max_groupping_dist_list = max_groupping_dist
            else:
                max_groupping_dist_list = [max_groupping_dist] if max_groupping_dist is not None else [3, 5, 8]

            if show_process:
                output[pr] = cv2.cvtColor(self.work_imgs[pr], cv2.COLOR_GRAY2BGR)  # Convert to color for visualization

            # Search for possible grids using all parameter combinations
            params = list(product(min_line_length_list, max_line_gap_list, max_groupping_dist_list))
            for line_length, line_gap, groupping_dist in params:

                # Detect horizontal and vertical lines
                v_lines, h_lines = self.find_lines(pr, line_length, line_gap)
                # Create a list of all possible x and y coordinates
                # Extract x values from vertical lines and y values from horizontal lines
                x_values = [x1 for x1, y1, x2, y2 in v_lines] + [x2 for x1, y1, x2, y2 in v_lines]
                y_values = [y1 for x1, y1, x2, y2 in h_lines] + [y2 for x1, y1, x2, y2 in h_lines]
                # Unify close values, replacing them with their mean
                x_values = unify_pixels(x_values, groupping_dist)
                y_values = unify_pixels(y_values, groupping_dist)
                # Determine possible distances between grid points
                dist_x = [x_values[i+1] - x_values[i] for i in range(len(x_values)-1)]
                dist_y = [y_values[i+1] - y_values[i] for i in range(len(y_values)-1)]
                all_sizes = unify_pixels(dist_x + dist_y, 0.5)
                all_sizes = [s for s in all_sizes if s > groupping_dist * 3]
                # Identify the best match for x_axis and y_axis 
                # using coordinate values and possible grid point distances
                best_match_x = self.find_axis(all_sizes, x_values, groupping_dist)
                best_match_y = self.find_axis(all_sizes, y_values, groupping_dist)
                x_axis = best_match_x['axis']
                y_axis = best_match_y['axis']

                # Build the list of valid grids (square-shaped and within size limits)
                valid_grid = (len(x_axis) == len(y_axis)) and (min_valid_n <= len(x_axis) <= max_valid_n)
                if valid_grid:    
                    self.all_grids.append({
                        'pre-proc': pr,
                        'in_final_set': False,
                        'label': -1,
                        'n': len(x_axis),
                        'height': height, 
                        'width': width,
                        'min_line_length': line_length, 
                        'max_line_gap': line_gap, 
                        'max_groupping_dist': groupping_dist,
                        'x_size': np.mean([x_axis[i+1] - x_axis[i] for i in range(len(x_axis) - 1)]),
                        'y_size': np.mean([y_axis[i+1] - y_axis[i] for i in range(len(y_axis) - 1)]),
                        'x_axis': x_axis,
                        'y_axis': y_axis,
                    })

                if show_process:
                    # Draw the detected lines, points, and grids on the processed images for visualization
                    for x1, y1, x2, y2 in h_lines:
                        cv2.line(output[pr], (x1, y1), (x2, y2), Color.ORGANGE, Grid.OUTPUT_LINE_THICKNESS)
                        cv2.circle(output[pr], (int(x1), int(y1)), Grid.OUTPUT_POINT_RADIUS, Color.ORGANGE, -1)
                        cv2.circle(output[pr], (int(x2), int(y2)), Grid.OUTPUT_POINT_RADIUS, Color.ORGANGE, -1)
                    for x1, y1, x2, y2 in v_lines:
                        cv2.line(output[pr], (x1, y1), (x2, y2), Color.RED, Grid.OUTPUT_LINE_THICKNESS)
                        cv2.circle(output[pr], (int(x1), int(y1)), Grid.OUTPUT_POINT_RADIUS, Color.RED, -1)
                        cv2.circle(output[pr], (int(x2), int(y2)), Grid.OUTPUT_POINT_RADIUS, Color.RED, -1)
                    for x in x_values:
                        for y in y_values:
                            cv2.circle(output[pr], (int(x), int(y)), Grid.OUTPUT_POINT_RADIUS, Color.GREEN, -1)
                    draw_grid(output[pr], x_axis, y_axis, frame=valid_grid, inplace=True)

        # choose the best grid among the valid ones
        best_n = 0
        best_label = None
        max_cluster_size = 0

        # make groups by value of n
        grids_by_n = {}
        for g in self.all_grids:
            grids_by_n.setdefault(g['n'], []).append(g)

        # apply clustering inside each group
        metric = 'euclidean'
        for n, group in grids_by_n.items():
            data = np.array([np.concatenate((g['x_axis'], g['y_axis'])) for g in group])
            eps = 1.0
            if len(group):
                # calculate the max distance between points to be considered one cluster
                eps = pdist(np.array([[0.0] * len(data[0]), [np.max([g['max_groupping_dist'] for g in group])] * len(data[0])]), metric)[0]
                # apply DBSCAN
                clustering = DBSCAN(eps=eps, min_samples=1, metric=metric).fit(data)
                labels = clustering.labels_
                # Count valid clusters
                label_counts = Counter(labels[labels != -1])
                # keep the labels for each grid
                for g, label in zip(group, labels):
                    g['label'] = label
                    g['cluster_size'] = label_counts[label] if label != -1 else 0
                # find the largest cluster in this group
                best_label_n = max(label_counts, key=lambda lbl: label_counts[lbl])
                cluster_size = label_counts[best_label_n]
                # If it's the largest cluster or with the largest n, update best_n and best_label
                if cluster_size > max_cluster_size or (cluster_size == max_cluster_size and n > best_n):
                    best_n = n
                    best_label = best_label_n
                    max_cluster_size = cluster_size

        # calculate the best grid as the mean value of all the 'best' ones
        self.n = best_n
        self.best_label = best_label
        self.max_cluster_size = max_cluster_size
        for grid in self.all_grids:
            grid['in_final_set'] = (grid['n'] == best_n) and (grid['label'] == best_label)
        self.x_axis, self.y_axis = [], []
        self.x_size, self.y_size = 0.0, 0.0
        if best_n > 0:
            self.x_axis = [
                int(np.mean([
                    grid['x_axis'][i] * self.width / grid['width'] 
                    for grid in self.all_grids 
                    if grid['in_final_set']
                ]) + 0.5) 
                for i in range(best_n)
            ]
            self.y_axis = [
                int(np.mean([
                    grid['y_axis'][i] * self.height / grid['height'] 
                    for grid in self.all_grids 
                    if grid['in_final_set']
                ]) + 0.5) 
                for i in range(best_n)
            ]
            self.x_size = float(np.mean([self.x_axis[i+1] - self.x_axis[i] for i in range(best_n - 1)]))
            self.y_size = float(np.mean([self.y_axis[i+1] - self.y_axis[i] for i in range(best_n - 1)]))

        if show_process:
            output['- Result -'] = draw_grid(self.image, self.x_axis, self.y_axis)

        return output

    def find_lines(self, pr, min_line_length, max_line_gap):
        """
        Detects horizontal and vertical lines in a given image using the Canny edge detector and Hough Line Transform.

        This function processes an image to detect lines and classifies them as horizontal or vertical.
        It applies the Canny edge detection algorithm to highlight edges in the image and then uses the probabilistic 
        Hough Line Transform to detect line segments. A line segment is classified as horizontal or vertical if its angle with
        respect to the x-axis or y-axis is within ±10°.

        :param pr (str): Key for the dictionary `self.work_imgs`. `self.work_imgs[pr]` is expected to be a preprocessed image.
        :param min_line_length (int): Minimum allowed length of a line segment to be detected.
        :param max_line_gap (int): Maximum allowed gap between line segments for them to be connected as a single line.

        :return: (tuple) A tuple `(v_lines, h_lines)` where:
                - v_lines (list): List of vertical line segments. Each line segment is represented as a list `[x1, y1, x2, y2]` 
                                  containing the starting and ending coordinates.
                - h_lines (list): List of horizontal line segments, also represented as `[x1, y1, x2, y2]`.
        """

        edges = cv2.Canny(self.work_imgs[pr], 50, 150)
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180, threshold=50, 
            minLineLength=min_line_length, 
            maxLineGap=max_line_gap
        )
        if lines is None:
            return [], []
        lines = [line[0] for line in lines]
        h_lines = []
        v_lines = []
        for line in lines:
            x1, y1, x2, y2 = line
            angle = np.arctan2(abs(y2 - y1), abs(x2 - x1)) * 180 / np.pi
            if angle <= 10:
                h_lines.append(line)
            elif angle >= 80:
                v_lines.append(line)
        return v_lines, h_lines
    

    def find_axis(self, all_sizes, all_values, max_groupping_dist):
        """
        Identifies the most likely grid axis (x or y) and cell size by matching coordinate values at regular intervals.

        For each possible cell size, this function takes one initial coordinate value from the list and tries to find other 
        values that match at regular intervals of that size. It tracks how many matches are found for each potential cell 
        size, and keeps the one that achieves the highest number of matches. The result is the best matching cell size and 
        a list of coordinates representing the detected grid lines (x or y).

        :param all_sizes (list of float): List of possible cell sizes to evaluate.
        :param all_values (list of float): Sorted list of coordinate values (x or y) representing potential grid line positions.
        :param max_groupping_dist (float): Maximum allowed distance to group a coordinate value as part of the grid 
                                           (i.e., tolerance for detecting near-regular intervals).
        
        :return: (dict) A dictionary with the following keys:
                - 'cell_size' (float): The cell size that produced the best match.
                - 'axis' (list of float): The list of coordinates that define the detected grid axis (x or y).
                - 'count' (int): The number of matches found for the best cell size.
        """
        best_match = {'cell_size': 0.0, 'axis': [], 'count': 0}
        # Iterate through each possible cell size
        for cell_size in all_sizes:
            # Try starting the grid at each coordinate value
            for i in range(len(all_values)-1):
                axis = [all_values[i]]
                next_value = axis[-1] + cell_size  # Calculate the expected next grid line position based on the cell size
                count = 0
                j = i + 1
                # Iterate through the remaining coordinate values to find matches at regular intervals
                while j < len(all_values):
                    if next_value < all_values[j] - max_groupping_dist:
                        # If the next expected grid line is smaller than the current value minus the allowed distance,
                        # it means no match was found, so append the expected value and calculate the next one.
                        axis.append(next_value)
                        next_value += cell_size
                    elif abs(next_value - all_values[j]) <= max_groupping_dist:
                        # If the current coordinate value matches the expected position within the allowed tolerance,
                        # it is considered part of the grid. Update the next expected value accordingly.
                        axis.append(all_values[j])
                        next_value = all_values[j] + cell_size
                        count += 1
                        j += 1
                    else:
                        # If no match, check the next coordinate value
                        j += 1
                # Keep the match with the highest count. In case of a tie, prefer the larger cell size.
                if count >= best_match['count']:
                    best_match['cell_size'] = cell_size
                    best_match['axis'] = axis
                    best_match['count'] = count
        return best_match


def unify_pixels(coords, max_distance):
    """
    Groups nearby pixel coordinates into clusters and replaces each cluster with its mean value.

    This function takes a list of pixel coordinates and merges them into unified clusters based on a specified maximum distance.
    If the difference between consecutive coordinates exceeds the given distance, a new cluster is started. Each cluster is 
    represented by the mean value of its coordinates.

    :param coords (list of float or int): List of pixel coordinates (e.g., x or y positions) to be unified.
    :param max_distance (float): Maximum allowed distance between consecutive coordinates for them to be considered part of the 
                                 same cluster.

    :return: (list of float) A list of unified pixel coordinates, where each value represents the mean of one cluster.
    """
    # If the input list has 1 or 0 coordinates, no unification is needed.
    if len(coords) <= 1:
        return coords

    # Sort the coordinates to ensure they are evaluated in ascending order.
    coords.sort()

    current, mean = [], coords[0]  # Initialize a temporary cluster and set the first mean to the first coordinate.
    ret = []  # List to store the unified coordinates (cluster means).

    # Iterate over each coordinate and group nearby ones based on the max_distance threshold.
    for coord in coords:
        if abs(coord - mean) > max_distance:
            # If the current coordinate is too far from the cluster's mean, finalize the current cluster.
            ret.append(mean)
            current = []  # Start a new cluster.
        current.append(coord)  # Add the current coordinate to the ongoing cluster.
        mean = np.mean(current)  # Update the mean of the cluster with the new coordinate.

    ret.append(mean)  # Append the mean of the last cluster to the result list.
    return ret


def draw_grid(img, x_axis, y_axis, frame=None, inplace=False):
    """
    Draws a grid of points at the intersections of specified x and y coordinates on the given image. 
    Optionally, draws a rectangle around the grid to frame it.

    :param img (numpy.ndarray): The input image on which the grid will be drawn. Can be grayscale or color.
    :param x_axis (list of float or int): List of x-coordinates where vertical grid lines intersect the y-axis.
    :param y_axis (list of float or int): List of y-coordinates where horizontal grid lines intersect the x-axis.
    :param frame (bool, optional): Controls whether a rectangle is drawn around the grid. If `None` (default), 
                                   the frame is drawn only for square grids. If explicitly set to `True`, 
                                   the frame is always drawn. If set to `False`, it is never drawn.
    :param inplace (bool, optional): If True, the grid is drawn directly on the input image. If False (default), 
                                     a copy of the image is created, and the grid is drawn on the copy.

    :return: (numpy.ndarray) The image with the drawn grid.
    """
    # If inplace is True, draw directly on the input image; otherwise, create a copy.
    out = img if inplace else img.copy()
    # Convert grayscale images to BGR color to allow colored grid visualization.
    if len(out.shape) == 2:  
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    # Draw blue circles at the intersections of x and y coordinates.
    for x in x_axis:
        for y in y_axis:
            cv2.circle(out, (int(x), int(y)), Grid.OUTPUT_POINT_RADIUS, Color.BLUE, thickness=-1)
    # Decide whether to draw a frame if frame=None, based on grid shape (square grid).
    frame = len(x_axis) == len(y_axis) and len(x_axis) > 1 if frame is None else frame
    # If `frame` is True, draw a rectangle around the outer boundaries of the grid.
    if frame:
        cv2.rectangle(
            out, 
            (int(min(x_axis)), int(min(y_axis))),  # Top-left corner of the rectangle.
            (int(max(x_axis)), int(max(y_axis))),  # Bottom-right corner of the rectangle.
            color=Color.BLUE, 
            thickness=Grid.OUTPUT_LINE_THICKNESS * 5
        )
    return out


def imshow(image):
    cv2.imshow("Imagen", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

