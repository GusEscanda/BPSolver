from itertools import product
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from collections import Counter
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

class Color:
    # BGR colors
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
        # handle resize parameters defaults
        width, height = self.width, self.height
        if resize_width or resize_height or resize_factor:
            # calculate width
            if resize_width:
                width = resize_width
            elif resize_factor:
                width = int((self.width * resize_factor) + 0.5)
            elif resize_height:
                width = int((self.width * resize_height / self.height) + 0.5)
            # calculate height
            if resize_height:
                height = resize_height
            elif resize_factor:
                height = int((self.height * resize_factor) + 0.5)
            elif resize_width:
                height = int((self.height * resize_width / self.width) + 0.5)

        # resize if needed
        image = self.image.copy()
        if width != self.width or height != self.height:
            method_w = cv2.INTER_CUBIC if width > self.width else cv2.INTER_AREA
            method_h = cv2.INTER_CUBIC if height > self.height else cv2.INTER_AREA
            method = method_w if method_h == method_w else cv2.INTER_LINEAR
            image = cv2.resize(image, (width, height), interpolation=method)

        # build the work images
        self.work_imgs = {}
        if len(image.shape) == 2:
            img_gray = image.copy()
        elif len(image.shape) == 3 and image.shape[2] == 3:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise Exception('Invalid image format')
        self.work_imgs['GRAY'] = img_gray

        for thr in threshold_values:
            _, binary = cv2.threshold(img_gray, thr, 255, cv2.THRESH_BINARY_INV)
            self.work_imgs[f'GRAY|BIN:{("000"+str(thr))[-3:]}'] = binary

        blurs = {}
        for pr in self.work_imgs:
            blur = cv2.GaussianBlur(self.work_imgs[pr], blur_ksize, 0)
            blurs[f"{pr}|BLUR"] = blur
        self.work_imgs = {**self.work_imgs, **blurs}


    def find_grid(
            self,
            min_line_length=None,       # minimum length for a line to be considered
            max_line_gap=None,          # maximo tamaño de corte que puede haber en la continuidad de una linea
            max_groupping_dist=None,    # distancia dentro de la cual dos puntos o lineas de consideran el mismo
            min_valid_n=3,              # minima cantidad de puntos por lado
            max_valid_n=16,             # maxima cantidad de puntos por lado
            show_process=False          # draw parcial results in the work images
        ):

        if not self.work_imgs:
            self.preprocess_image()
        
        output = {} if show_process else None

        # for all the preprocesed images find valid grids
        self.all_grids = []
        for pr in self.work_imgs:

            # set the parameters to search this image for grids
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
                output[pr] = cv2.cvtColor(self.work_imgs[pr], cv2.COLOR_GRAY2BGR)  # Convertir a color para visualizar

            # for all the parameters look for posible grids
            params = list(product(min_line_length_list, max_line_gap_list, max_groupping_dist_list))
            for line_length, line_gap, groupping_dist in params:

                # find horizontal and vertical lines
                v_lines, h_lines = self.find_lines(pr, line_length, line_gap)
                # list all the posible values for the coord x and y
                # take the x values from the vertical lines and the y values from the horizontal lines
                x_values = [x1 for x1, y1, x2, y2 in v_lines] + [x2 for x1, y1, x2, y2 in v_lines]
                y_values = [y1 for x1, y1, x2, y2 in h_lines] + [y2 for x1, y1, x2, y2 in h_lines]
                # unify close values, make the close values exactly equal to their mean
                x_values = unify_pixels(x_values, groupping_dist)
                y_values = unify_pixels(y_values, groupping_dist)
                # get the posible distances between the points of the grid
                dist_x = [x_values[i+1] - x_values[i] for i in range(len(x_values)-1)]
                dist_y = [y_values[i+1] - y_values[i] for i in range(len(y_values)-1)]
                all_sizes = unify_pixels(dist_x + dist_y, 0.5)
                all_sizes = [s for s in all_sizes if s > groupping_dist * 3]
                # find the best match for an x_axis and a y_axis 
                # using the values of the coordinates and the posible distances between dots
                best_match_x = self.find_axis(all_sizes, x_values, groupping_dist)
                best_match_y = self.find_axis(all_sizes, y_values, groupping_dist)
                x_axis = best_match_x['axis']
                y_axis = best_match_y['axis']

                # build the list of valid grids (squared and with size between the limits)
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
                    # draw on the work images to show the findings of the process
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
        Para cada posible tamaño de celda, toma un valor de coordenada (x o y) y busca coincidencias con los otros valores a intervalos
        regulares de ese tamaño. Se queda con el tamaño que mas coincidencias logra y devuelve tambien la lista de las coordenadas de esas
        coincidencias, que serán las coordenadas de la grilla (de x o y)
        """
        best_match = {'cell_size': 0.0, 'axis': [], 'count': 0}
        for cell_size in all_sizes:
            for i in range(len(all_values)-1):
                axis = [all_values[i]]
                next_value = axis[-1] + cell_size
                count = 0
                j = i + 1
                while j < len(all_values):
                    if next_value < all_values[j] - max_groupping_dist:
                        axis.append(next_value)
                        next_value += cell_size
                    elif abs(next_value - all_values[j]) <= max_groupping_dist:  # si unif[j] esta a una cantidad de saltos regulares de longitud cell_size desde inic
                        axis.append(all_values[j])
                        next_value = all_values[j] + cell_size
                        count += 1
                        j += 1
                    else:
                        j += 1
                if count >= best_match['count']:  # a igual caqntidad de coincidencias me quedo con el cell_size mayor
                    best_match['cell_size'] = cell_size
                    best_match['axis'] = axis
                    best_match['count'] = count
        return best_match


def unify_pixels(coords, max_distance):
    if len(coords) <= 1:
        return coords
    coords.sort()
    current, mean = [], coords[0]
    ret = []
    for coord in coords:
        if abs(coord - mean) > max_distance:
            ret.append(mean)
            current = []
        current.append(coord)
        mean = np.mean(current)
    ret.append(mean)
    return ret


def draw_grid(img, x_axis, y_axis, frame=None, inplace=False):
    out = img if inplace else img.copy()
    if len(out.shape) == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)  # Convertir a color para visualizar
    for x in x_axis:
        for y in y_axis:
            cv2.circle(out, (int(x), int(y)), Grid.OUTPUT_POINT_RADIUS, Color.BLUE, thickness=-1)
    frame = len(x_axis) == len(y_axis) and len(x_axis) > 1 if frame is None else frame
    if frame:
        cv2.rectangle(
            out, 
            (int(min(x_axis)), int(min(y_axis))), 
            (int(max(x_axis)), int(max(y_axis))), 
            color=Color.BLUE, thickness=Grid.OUTPUT_LINE_THICKNESS * 5
        )
    return out


def imshow(image):
    cv2.imshow("Imagen", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

