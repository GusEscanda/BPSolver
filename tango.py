from itertools import product, chain
from collections import Counter
import re
import numpy as np

from skimage.feature import hog
from sklearn.cluster import DBSCAN
import cv2

from BPSolver import BoardPuzzle
from grids import Grid

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

class Tango(BoardPuzzle):

    class Move:
        def __init__(self, coord: tuple, tile: str):
            self.coord = coord
            self.tile = tile


    def __init__(self, source = None, **addit_params):
        """
        Represents the Tango game from LinkedIn, implementing the abstract class BoardPuzzle.

        This class takes an unsolved screenshot of the game, searches for the board grid in the image, or directly 
        receives a Grid object if the grid has already been detected. It initializes the necessary data structures 
        for use by the BPSolver class.

        :param source: Either a string representing the image filename to be processed or an object of type Grid. 
                    If a string is provided, a Grid object will be created and initialized by detecting the grid 
                    in the image. If a Grid object is passed directly, it will be used as is if it already contains 
                    a detected grid (Grid.n > 0). Otherwise, it will attempt to detect the grid using the methods 
                    Grid.preprocess_image and Grid.find_grid.
        :type source: str or Grid
        :param addit_params: Additional keyword arguments that will be passed to the appropriate methods of the Grid object. 
                            Parameters related to image preprocessing (e.g., `resize_width`, `threshold_values`) will be 
                            sent to `Grid.preprocess_image`, while parameters related to grid detection (e.g., `min_line_length`, 
                            `max_groupping_dist`) will be sent to `Grid.find_grid`.
        :type addit_params: dict-like (keyword arguments)
        """
        self.n = 0
        self.board = None
        self.ori_board = None
        self.ver_constraints = None
        self.hor_constraints = None
        self.str_board = None
        self.moves_by_level = dict()
        self.order_within_level = 0
        self.is_solved = False
        self.is_invalid = False
        self.image = None
        self.constraint_templates = None
        self.x_axis = None
        self.y_axis = None
        self.size = None
        self.msg = ''
        if source is None:
            return
        
        try:
            if isinstance(source, str):
                # set a Grid object with the given image file
                grid = Grid(image_path=source)
                if grid.image is None:
                    self.msg = 'Could not read image file'
                    return
            elif isinstance(source, Grid):
                # use the Grid object provided
                grid = source
            else:
                self.msg = 'Source must be a filename or a Grid object'
                return
                        
            if grid.n == 0 and grid.image is not None:
                # find the grid in the image
                grid.preprocess_image(**{p: addit_params[p] for p in addit_params if p in {
                    'resize_width', 
                    'resize_height', 
                    'resize_factor', 
                    'threshold_values', 
                    'blur_ksize'
                }})
                grid.find_grid(
                    **{p: addit_params[p] for p in addit_params if p in {
                        'min_line_length', 
                        'max_line_gap', 
                        'max_groupping_dist', 
                    }},
                    min_valid_n=7,
                    max_valid_n=7
                )
            if grid.n != 7:
                self.msg = 'No valid (6x6) grid detected'
                return
            
            if 'constraint_templates' in addit_params:
                constraint_templates = addit_params['constraint_templates']
            elif 'eq_filename' in addit_params and 'x_filename' in addit_params:
                constraint_templates = self.load_templates_from_files(
                    eq_filename = addit_params['eq_filename'],
                    x_filename = addit_params['x_filename']
                )
            else:
                constraint_templates = self.create_synthetic_templates()

            # load the board from the grid
            self.msg = self.load_from_grid(grid, constraint_templates)
            return

        except Exception as ex:
            self.msg = str(ex)            
            return


    @staticmethod
    def create_synthetic_templates(font_scale=[0.7, 0.8], thickness=[1, 2]):
        templates = {}
        symbols = ['=', 'x']
        for symbol, scale, thick in product(symbols, font_scale, thickness):
            # Fondo blanco, símbolo negro
            img = np.ones((100, 100), dtype=np.uint8) * 255
            cv2.putText(img, symbol, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, scale, 0, thick)

            # Recortar el simbolo, encontrar filas y columnas donde hay al menos un píxel negro
            mask = img < 255
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            img = img[ymin-2:ymax+3, xmin-2:xmax+3] # dejo un margen de 2 pixeles alrededor de la imagen

            templates[f'{symbol}-{scale}-{thick}'] = img
        return templates

    @staticmethod
    def load_templates_from_files(eq_filename, x_filename, size=20):
        templates = {}
        for symbol, filename in [('=', eq_filename), ('x', x_filename)]:
            image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            height, width = image.shape[:2]
            version = 0
            for row, col in product(range(height//size), range(width//size)):
                
                img = image[row*size:(row+1)*size, col*size:(col+1)*size]
                if np.all(img == 255):
                    continue

                # Recortar el simbolo, encontrar filas y columnas donde hay al menos un píxel negro
                mask = img < 255
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                ymin, ymax = np.where(rows)[0][[0, -1]]
                xmin, xmax = np.where(cols)[0][[0, -1]]
                img = img[ymin-2:ymax+3, xmin-2:xmax+3] # dejo un margen de 2 pixeles alrededor de la imagen

                version += 1
                templates[f'{symbol}-{version}'] = img

        return templates

    @staticmethod
    def find_symbol(img, templates, scales=[0.9, 1.0, 1.1], threshold=0.8):
        best_val = 0
        best_symbol = '-'

        for symbol, tmpl in templates.items():
            for scale in scales:
                scaled_tmpl = cv2.resize(tmpl, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                if scaled_tmpl.shape[0] > img.shape[0] or scaled_tmpl.shape[1] > img.shape[1]:
                    continue  # no comparar si el template es más grande que la imagen

                result = cv2.matchTemplate(img, scaled_tmpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)

                if max_val > threshold and max_val > best_val:
                    best_val = max_val
                    best_symbol = symbol
                
                inverted_tmpl = 255 - scaled_tmpl

                result = cv2.matchTemplate(img, inverted_tmpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)

                if max_val > threshold and max_val > best_val:
                    best_val = max_val
                    best_symbol = symbol

        return best_symbol[0]

    def load_from_grid(self, grid: Grid, constraint_templates):
        """
        Loads and initializes the board based on the grid detected in the provided Grid object.
        
        This method extracts key information from the given Grid object, including the image, grid axis positions, 
        and cell size. Uses HOG, matchTemplate and DBSCAN to detect all the objects in the image and initializes the 
        internal structures of the board. If the grid is invalid (e.g., no grid detected or not a 6x6 grid), 
        it resets the object.

        :param grid: A Grid object containing the detected grid and its properties (image, axis positions, cell size).
        :type grid: Grid
        
        :return: A status message indicating whether the board was successfully loaded. Possible return values are:
                 - 'OK': The board was successfully loaded.
                 - 'No valid grid detected': The provided Grid object isn't a 6x6 grid
                 - 'No valid board detected': The grid was detected, but there are more than two types of tiles.
        :rtype: str
        """
        self.n = grid.n - 1  # n coordinates in each axis implies n-1 cells
        if self.n != 6:
            self.__init__()
            return 'No valid (6x6) grid detected'
        self.image = grid.image
        self.x_axis = grid.x_axis
        self.y_axis = grid.y_axis
        self.size = int(round(np.mean([grid.x_size, grid.y_size]), 0))
        self.constraint_templates = constraint_templates

        n = self.n

        # Diccionario para acumular features de cada celda en todas las versiones de imagen
        cell_features = {(row, col): [] for row, col in product(range(n), range(n))}

        uniformities = {}
        self.ver_constraints = np.full((n, n - 1), '', dtype=object)  # n-1 lineas divisorias horizontales en cada una de las n columnas
        self.hor_constraints = np.full((n, n - 1), '', dtype=object)  # n-1 lineas divisorias verticales en cada una de las n filas

        procs = set(g['pre-proc'] for g in grid.all_grids if g['in_final_set'])  # use the preprocessed images that were useful to find the grid
        for pr in procs:
            image = grid.work_imgs[pr]
            height, width = image.shape[:2]
            x_axis = np.array(grid.x_axis) * width / grid.width
            y_axis = np.array(grid.y_axis) * height / grid.height
            min_size = min(
                min([x_axis[i+1] - x_axis[i] for i in range(n)]),
                min([y_axis[i+1] - y_axis[i] for i in range(n)])
            )
            half_size = int(min_size / 2) - 10  # Subtract 10 pixels to avoid taking grid lines

            for row, col in product(range(n), range(n)):
                x_beg, x_end = int(x_axis[col]), int(x_axis[col+1])
                y_beg, y_end = int(y_axis[row]), int(y_axis[row+1])
                x, y = (x_beg + x_end) // 2, (y_beg + y_end) // 2
                
                cell = image[y-half_size : y+half_size, x-half_size : x+half_size]

                if row < n-1:
                    line = row
                    area = image[y_end-10 : y_end+10, x-half_size : x+half_size]
                    symbol = self.find_symbol(area, constraint_templates)
                    self.ver_constraints[col][line] += symbol
                if col < n-1:
                    line = col
                    area = image[y-half_size : y+half_size, x_end-10 : x_end+10]
                    symbol = self.find_symbol(area, constraint_templates)
                    self.hor_constraints[row][line] += symbol

                # Extract HOG features
                hog_features =  hog(cell, pixels_per_cell=(10, 10), cells_per_block=(3, 3), feature_vector=True)
                cell_features[(row, col)].append(hog_features)

                # Calulate uniformities (the label corresponding to the most uniform cell will be considered the 'blank' label)
                uniformities[(row, col)] = np.std(cell)

        # Promediar las features de cada celda entre todas las versiones de imagen
        final_features = []
        cell_positions = []
        for (row, col), features in cell_features.items():
            if features:
                final_features.append(np.mean(features, axis=0))
                cell_positions.append((row, col))

        # Clasificar con DBSCAN sobre las features unificadas
        dbscan = DBSCAN(eps=2, min_samples=1)
        labels = dbscan.fit_predict(final_features)

        if not (1 <= len(set(labels)) <= 3):
            return 'No valid board detected'

        # Find the most uniform cell and set the corresponding label as the 'empty' label
        empty_cell = min(uniformities, key=uniformities.get)
        empty_label = labels[cell_positions.index(empty_cell)]
        label_dict = {int(empty_label): ' '}
        next_label = 1

        # Asignar los clusters a cada celda del tablero
        self.board = np.array([[' '] * n] * n)
        for (row, col), label in zip(cell_positions, labels):
            if int(label) not in label_dict:
                label_dict[int(label)] = str(next_label)
                next_label += 1
            self.board[row, col] = label_dict[int(label)]

        self.ori_board = self.board.copy()

        # Elegir entre los constraints los que mas se repiten
        self.ver_constraints[:] = np.vectorize(lambda s: max(set(s), key=s.count))(self.ver_constraints) 
        self.hor_constraints[:] = np.vectorize(lambda s: max(set(s), key=s.count))(self.hor_constraints) 

        self.str_board = self.board2strings()
        return 'OK'
    
    def board2strings(self):
        str_board = {}
        for row in range(self.n):
            str_board[(row, None)] = ''
            for i in range(self.n-1):
                str_board[(row, None)] += self.board[row, i] + self.hor_constraints[row, i]
            str_board[(row, None)] += self.board[row, -1]
        for col in range(self.n):
            str_board[(None, col)] = ''
            for i in range(self.n-1):
                str_board[(None, col)] += self.board[i, col] + self.ver_constraints[col, i]
            str_board[(None, col)] += self.board[-1, col]
        return str_board

    def draw_solution(self, image = None, inplace = False):
        """
        Draws the solution of the board puzzle on the given image.

        This method visually represents the puzzle's solution by drawing the board's tiles and constraints. 
        Each tile and constraint found in the original board is represented by a different color circle (outlined
        so it's posible to see the object in the original image).
        Each tile placed by the solver is represented with a circle with the same color code, if the puzzle is marked as solved
        these circles are drawn with solid colors, otherwise they are outlined.

        :param image: Optional. The image on which to draw the solution. If not provided, it defaults to 
                      `self.image`. If both `image` and `self.image` are `None`, the method returns `None`.
        :param inplace: Optional. A boolean indicating whether to modify the given image in-place. 
                        If `True`, the drawing is applied directly to the input image. If `False` (default), a copy 
                        of the image is created and modified instead.

        :return: The image with the solution drawn on it. If no valid image is provided, returns `None`.
        """

        image = self.image if image is None else image
        if image is None or self.board is None:
            return None if image is None else (image if inplace else image.copy())
        out = image if inplace else image.copy()
        if len(out.shape) == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        colors = {
            '1': Color.GREEN,
            '2': Color.RED,
            '=': Color.CYAN,
            'x': Color.ORGANGE
        }
        n = 6
        tile_thickness = 12
        tile_radius = int(self.size * 0.32)
        constraint_thicnkess = 5
        constraint_radius = int(self.size * 0.2)
        # draw the tiles
        for row in range(n):
            for col in range(n):
                x = int(self.x_axis[col] + self.x_axis[col+1]) // 2
                y = int(self.y_axis[row] + self.y_axis[row+1]) // 2
                if self.board[row, col] != ' ':
                    thickness = -1 if self.is_solved and self.ori_board[row, col] == ' ' else tile_thickness
                    cv2.circle(out, (x, y), tile_radius, colors[self.board[row,col]], thickness=thickness)
        # draw the vertical constraints
        for col in range(n):
            x = int(self.x_axis[col] + self.x_axis[col+1]) // 2
            for line in range(n-1):
                y = int(self.y_axis[line+1])
                if self.ver_constraints[col, line] != '-':
                    cv2.circle(out, (x, y), constraint_radius, colors[self.ver_constraints[col,line]], thickness=constraint_thicnkess)
        # draw the horizontal constraints
        for row in range(n):
            y = int(self.y_axis[row] + self.y_axis[row+1]) // 2
            for line in range(n-1):
                x = int(self.x_axis[line+1])
                if self.hor_constraints[row, line] != '-':
                    cv2.circle(out, (x, y), constraint_radius, colors[self.hor_constraints[row,line]], thickness=constraint_thicnkess)
        return out
    
    def solved(self):
        """
        Checks if the puzzle is solved.

        :return: True if the puzzle is solved; otherwise, False.
        """
        return self.is_solved and self.board is not None

    def is_dead_end_or_invalid(self):
        """
        Determines whether the current board state is invalid or leads to a dead end.

        :return: True if the board state is invalid or leads to a dead end; otherwise, False.
        """
        return self.is_invalid and self.board is not None

    def candidate_moves(self, level: int):
        """
        Generates an iterable of possible moves from the current board state, ordered to maximize the chances 
        of finding a solution quickly.

        :param level: The current recursive depth, useful for undoing all moves if a dead end is reached.
        :return: An iterable of possible moves from the current position.
        """
        if self.board is None:
            return []
        # find the row or column with a tile qty closest to 3 but not equal to 3 and try that tile in each blank space
        best_tile, best_other, best_vector = None, None, None
        best_tc = {' ': 0, '1': -1, '2': -1, '-': -1}
        for vector_id in self.str_board:
            tc = self.tiles_count(vector_id)
            for tile, other in [('1', '2'), ('2', '1')]:
                if tc[tile] > 3:
                    self.is_invalid = True
                    return []
                if tc[tile] == 3:
                    continue
                if tc[tile] > best_tc[tile] or (tc[tile] == best_tc[tile] and tc['-'] < best_tc['-']):
                    best_tc, best_vector = tc, vector_id
                    best_tile, best_other = tile, other
        if best_tc is None or best_tc[' '] == 0:
            return []
        posic = self.str_find(best_vector, ' ')
        coord = self.posic2coord(best_vector, posic)
        moves = [self.Move(coord, best_tile), self.Move(coord, best_other)]
        return moves
    
    def str_find(self, vector_id, s):
        match = re.search(s, self.str_board[vector_id])
        return match.start() if match else None

    def make_move(self, move: Move, level: int):
        """
        Executes a move and records metadata needed to undo this and other moves at the same recursive level.

        :param move: An object representing the move, such as coordinates, piece type, or character.
        :param level: The current recursive depth, used to undo moves if a dead end is reached.
        """
        if self.board[move.coord] == move.tile:
            return
        # store in moves_by_level the coord of the move and the previous value on the board
        if level not in self.moves_by_level:
            self.moves_by_level[level] = []
            self.order_within_level = 0
        self.order_within_level += 1
        self.moves_by_level[level].append({
            'coord': move.coord,
            'tile': move.tile,
            'prev_tile': self.board[move.coord],
            'order': self.order_within_level
        })
        # make the move
        self.board[move.coord] = move.tile
        self.update_str_board(move.coord, move.tile)

    @staticmethod
    def posic2coord(vector_id, posic):
        row, col = vector_id
        return (row, posic//2) if row is not None else (posic//2, col)

    def tiles_count(self, vector_id):
        """
        Counts the number of tiles of each type ('1', '2', and ' ') in the specified row or column and performs checks on its validity.
        Also counts the number or constraints 'x' and '=' in the specified row or column

        :param row: The row number to analyze or None if a column is specified
        :param col: The column number to analyze or None if a row is specified
        
        :return: A dictionary containing counters for the three types of tiles ('1', '2', and ' ') and the two types of contraint ('x' and '=').
        """
        counter = Counter(self.str_board[vector_id])
        tc = {k: counter.get(k, 0) for k in '12 -=x'}
        self.is_invalid = self.is_invalid or tc['1'] > 3 or tc['2'] > 3
        return tc


    def check_and_move(self, level, vector_id, mask, changes):
        changed = False
        for _ in ('forth', 'back'):
            posic = self.str_find(vector_id, mask)
            if posic is not None:
                for i in range(len(mask)):
                    if mask[i] != changes[i]:
                        coord = self.posic2coord(vector_id, posic+i)
                        self.make_move(self.Move(coord, changes[i]), level)
                        changed = True
            mask, changes = mask[::-1], changes[::-1]  # try mirroring the mask
        return changed


    def deduct_enforce_constraints(self, level: int):
        """
        Marks all empty cells in a zone with an 'X' if there is already a queen placed in that zone.

        This method iterates through each zone on the board. If a zone contains exactly one queen and at least 
        one empty cell, it marks all remaining empty cells in that zone as blocked ('X').

        :param level: The current recursive depth, used to undo moves if a dead end is reached.

        :return: True if any change was made in the board; otherwise, False.
        """
        changed = False
        space = ' '
        complete, invalid = True, False
        for vector_id in self.str_board:
            tc = self.tiles_count(vector_id)
            complete = complete and tc[' '] == 0
            for tile, other in [('1', '2'), ('2', '1')]:
                invalid = invalid or tc[tile] > 3
                invalid = invalid or (self.str_find(vector_id, f'{tile}.{tile}.{tile}') is not None)  # 1 1 1  =>  invalid
                invalid = invalid or f'{tile}={other}' in self.str_board[vector_id]                   # 1=2  =>  invalid
                invalid = invalid or f'{tile}x{tile}' in self.str_board[vector_id]                    # 1x1  =>  invalid
                if invalid:
                    self.is_invalid = True
                    return changed
                if tc[' '] == 0:
                    continue
                
                changed = self.check_and_move(level, vector_id, f'{tile}.{tile}.{space}', f'{tile}.{tile}.{other}') or changed

                changed = self.check_and_move(level, vector_id, f'{tile}.{space}.{tile}', f'{tile}.{other}.{tile}') or changed

                changed = self.check_and_move(level, vector_id, f'{tile}={space}', f'{tile}={tile}') or changed

                changed = self.check_and_move(level, vector_id, f'{tile}x{space}', f'{tile}x{other}') or changed

                changed = self.check_and_move(level, vector_id, f'{tile}.{space}={space}', f'{tile}.{other}={other}') or changed

                if tc[tile] >= 2:
                    changed = self.check_and_move(level, vector_id, f'{space}={space}', f'{other}={other}') or changed
                    changed = self.check_and_move(level, vector_id, f'{space}x{space}x{space}', f'{other}x{tile}x{other}') or changed

                if tc[tile] == 3:
                    changed = self.check_and_move(
                        level, 
                        vector_id, 
                        self.str_board[vector_id], 
                        self.str_board[vector_id].replace(space, other)
                    ) or changed

        if complete:
            self.is_solved = True
        return changed

    def make_deduct_moves(self, level: int):
        """
        Executes all mandatory moves based on the current board state. These moves are labeled with the 
        current recursive level, allowing them to be undone if they lead to an invalid state or dead end.

        If a dead end is reached, clean_level() will remove all moves marked with the current level.

        :param level: The current recursive depth, used to undo moves if a dead end is encountered.
        """

        if self.board is None:
            return
        changed = True
        while changed and not self.is_invalid and not self.is_solved:
            changed = self.deduct_enforce_constraints(level)

    def update_str_board(self, coord, tile):
        row, col = coord
        s = self.str_board[(row, None)]
        self.str_board[(row, None)] = s[:col*2] + tile + s[col*2+1:]
        s = self.str_board[(None, col)]
        self.str_board[(None, col)] = s[:row*2] + tile + s[row*2+1:]


    def clean_level(self, level: int):
        """
        Removes all moves made by make_move() and make_deduct_moves() at the specified recursive level.

        :param level: The recursive level used to undo all moves that led to a dead end.
        """
        self.moves_by_level[level].reverse()

        for m in self.moves_by_level[level]:
            self.board[m['coord']] = m['prev_tile']
            self.update_str_board(m['coord'], m['prev_tile'])

        self.moves_by_level.pop(level)
        self.is_invalid = False

