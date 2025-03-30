import numpy as np

from sklearn.cluster import KMeans
import cv2

from BPSolver import BoardPuzzle
from grids import Grid


class Queens(BoardPuzzle):

    class Move:
        def __init__(self, coord: tuple, tile: str):
            self.coord = coord
            self.tile = tile

    def __init__(self, source = None, **addit_params):
        """
        Represents the Queens game from LinkedIn, implementing the abstract class BoardPuzzle.

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
        self.board = None
        self.zone_matrix = None
        self.zone_colors = None
        self.zone_lists = None
        self.zone_queens = dict()
        self.rows, self.cols = 0, 0
        self.moves_by_level = dict()
        self.order_within_level = 0
        self.is_solved = False
        self.is_invalid = False
        self.image = None
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
                grid.find_grid(**{p: addit_params[p] for p in addit_params if p in {
                    'min_line_length', 
                    'max_line_gap', 
                    'max_groupping_dist', 
                    'min_valid_n', 
                    'max_valid_n'
                }})
            if grid.n == 0:
                self.msg = 'No grid detected'
                return
            
            # load the board from the grid
            self.msg = self.load_from_grid(grid)
            return

        except Exception as ex:
            self.msg = str(ex)            
            return

    def load_from_grid(self, gr: Grid):
        """
        Loads and initializes the board based on the grid detected in the provided Grid object.
        
        This method extracts key information from the given Grid object, including the image, grid axis positions, 
        and cell size. It computes the average colors for a sub-region of each cell, groups the colors into clusters, 
        and initializes the internal structures of the board. If the grid is invalid (e.g., no grid detected or 
        insufficient distinct colors), it resets the object.

        :param gr: A Grid object containing the detected grid and its properties (image, axis positions, cell size).
        :type gr: Grid
        
        :return: A status message indicating whether the board was successfully loaded. Possible return values are:
                 - 'OK': The board was successfully loaded.
                 - 'No grid detected': The provided Grid object has no detected grid (`gr.n == 0`).
                 - 'No valid board detected': The grid was detected, but there are not enough distinct colors 
                   to distinguish between the expected zones.
        :rtype: str
        """
        if gr.n == 0:
            self.__init__()
            return 'No grid detected'
        self.image = gr.image
        self.x_axis = gr.x_axis
        self.y_axis = gr.y_axis
        self.size = int(round(np.mean([gr.x_size, gr.y_size]), 0))
        # Compute avg colors for a sub-zone in each cell
        n = gr.n - 1
        avg_colors = np.zeros((n, n, 3), dtype=np.uint8)
        x_shift_beg = max(int(round(gr.x_size * 0.2, 0)), 5)
        x_shift_end = max(int(round(gr.x_size * 0.3, 0)), 9)
        y_shift_beg = max(int(round(gr.y_size * 0.2, 0)), 5)
        y_shift_end = max(int(round(gr.y_size * 0.3, 0)), 9)
        for row in range(n):
            for col in range(n):
                cell_x, cell_y = gr.x_axis[col], gr.y_axis[row]
                sub_cell = gr.image[
                    cell_y + y_shift_beg : cell_y + y_shift_end, 
                    cell_x + x_shift_beg : cell_x + x_shift_end
                ]
                avg_colors[row, col] = np.mean(sub_cell, axis=(0, 1))
        # Group the colors
        avg_colors = avg_colors.reshape(-1, 3)
        unique_colors = np.unique(avg_colors, axis=0)
        if len(unique_colors) < n:
            self.__init__()
            return f'No valid board detected ({n}x{n} grid and only {len(unique_colors)} colors found)'
        clusters = KMeans(n_clusters=n, random_state=0)
        labels = clusters.fit_predict(avg_colors)
        unique_labels = np.unique(labels)
        if len(unique_labels) != n:
            self.__init__()
            return f'No valid board detected ({n}x{n} grid but {len(unique_labels)} colors found)'
        self.zone_matrix = labels.reshape(n, n)
        self.zone_colors = np.array([avg_colors[labels == label].mean(axis=0) for label in unique_labels], dtype=np.uint8)
        # Init other board structures
        self.board = np.full(self.zone_matrix.shape, " ", dtype=object)
        self.rows, self.cols = self.zone_matrix.shape
        self.zone_lists_init()
        return 'OK'

    def zone_lists_init(self):
        """
        Initializes the zone-based lists and queen tracking for the board.

        This method creates lists of positions for each zone in the board, based on the zone matrix. 
        A zone is defined either by a unique color cluster (inferred from the clustering algorithm) 
        or by structural components like rows and columns. It also initializes the tracking of queen 
        placements in each zone, setting them to `None` by default.

        The created zones include:
        - Color-based zones: Each distinct zone in `self.zone_matrix` is mapped to a list of cell coordinates 
          that belong to that zone.
        - Row-based zones: Each row is mapped to a list of cell coordinates in that row.
        - Column-based zones: Each column is mapped to a list of cell coordinates in that column.

        Additionally, it initializes `self.zone_queens`, which tracks the position of the queen (if any) in each zone.

        :return: None
        """
        self.zone_lists = dict()
        for (row, col), value in np.ndenumerate(self.zone_matrix):
            if f'zone {value}' not in self.zone_lists:
                self.zone_lists[f'zone {value}'] = []
            self.zone_lists[f'zone {value}'].append((row, col))
        for row in range(self.rows):
            self.zone_lists[f'row {row}'] = [(row, c) for c in range(self.cols)]
        for col in range(self.cols):
            self.zone_lists[f'col {col}'] = [(r, col) for r in range(self.rows)]
        for zone in self.zone_lists:
            self.zone_queens[zone] = None

    def draw_solution(self, image = None, inplace = False):
        """
        Draws the solution of the board puzzle on the given image.

        This method visually represents the puzzle's solution by drawing the board's zones and any queens placed 
        on the board. Each cell is filled with its corresponding zone color, and if a queen ('Q') is placed in a 
        cell, it is drawn as a circle. The queen is filled with a solid color if the puzzle is marked as solved, 
        otherwise, it is outlined.

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
        n = self.board.shape[0]
        radius = int(self.size * 0.3)
        out = image if inplace else image.copy()
        for row in range(n):
            for col in range(n):
                x = int(self.x_axis[col] + self.x_axis[col+1]) // 2
                y = int(self.y_axis[row] + self.y_axis[row+1]) // 2
                color = tuple(self.zone_colors[self.zone_matrix[row, col]].tolist())
                cv2.rectangle(out, (x-radius, y-radius), (x+radius, y+radius), color=color, thickness=-1)
                if self.board[row, col] == 'Q':
                    cv2.circle(out, (x, y), radius, (0, 0, 0), thickness=(-1 if self.is_solved else 2))
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
        # find the zone with min blank spaces (and no queen inside)
        zone, blank_qty = '', self.rows * self.cols + 1
        for z in self.zone_lists:
            tc = self.tiles_count(z)
            if tc[' '] <= blank_qty and tc['Q'] == 0:
                zone, blank_qty = z, tc[' ']
        ret = []
        if not zone:  # invalid or solved puzzle
            return ret
        # for each blank space, try a queen, then an X
        for coord in self.zone_lists[zone]:
            if self.board[coord] != ' ':
                continue
            ret.append(self.Move(coord, 'Q'))
            ret.append(self.Move(coord, 'X'))  # if the queen led to a dead end, then mark the cell with an X
        return ret

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

    def tiles_count(self, zone):
        """
        Counts the number of tiles of each type ('Q', 'X', and ' ') in the specified zone and performs checks on its validity.

        This method iterates through the tiles in the given zone and keeps track of the count of queens ('Q'), blocked cells ('X'), 
        and empty cells (' '). It also stores the position of the last tile of each type encountered. If the zone contains more 
        than one queen or has no empty cells left (all tiles are 'X'), the puzzle is marked as invalid.

        :param zone: The zone to analyze and count tiles in.
        
        :return: A dictionary containing:
            - Counters for the three types of tiles ('Q', 'X', and ' ').
            - The position of the last tile of each type ('last <Q>', 'last < >', 'last <X>').
        """
        ret = {'Q': 0, 'X': 0, ' ': 0, 'last <Q>': None, 'last < >': None, 'last <X>': None}
        for coord in self.zone_lists[zone]:
            ret[self.board[coord]] += 1
            ret[f'last <{self.board[coord]}>'] = coord
        self.zone_queens[zone] = ret['last <Q>']
        self.is_invalid = self.is_invalid or (ret['X'] == len(self.zone_lists[zone])) or (ret['Q'] > 1)
        return ret

    def zone_limits(self, zone):
        """
        Returns the boundaries of the current zone, i.e., the minimum and maximum row numbers and the minimum
        and maximum column numbers of all blank cells in a zone (if any).

        The return value is a dict with this structure:
            { 'min': (minimum row number, minimum col number),
              'max': (maximum row number, maximum col number) }

        :param zone: the zone to process
        :return: a structure as described above, or an empty dict
        """
        ret = dict()
        any_blank = any(self.board[coord] == ' ' for coord in self.zone_lists[zone])
        if any_blank:
            ret['min'] = (
                min([coord[0] for coord in self.zone_lists[zone] if self.board[coord] == ' ']),
                min([coord[1] for coord in self.zone_lists[zone] if self.board[coord] == ' '])
            )
            ret['max'] = (
                max([coord[0] for coord in self.zone_lists[zone] if self.board[coord] == ' ']),
                max([coord[1] for coord in self.zone_lists[zone] if self.board[coord] == ' '])
            )
        return ret

    def deduct_one_place_left(self, level: int):
        """
        Places a queen in any row, column, or zone that has only one possible position left.

        This method checks each row, column, and zone on the board. If any of them has just one empty cell 
        remaining, it places a queen in that cell.

        :param level: The current recursive depth, used to undo moves if a dead end is reached.

        :return: True if any change was made in the board; otherwise, False.
        """
        changed = False
        for zone in self.zone_lists:
            tc = self.tiles_count(zone)
            if self.is_invalid:
                break
            if tc[' '] == 1 and tc['Q'] == 0:
                self.make_move(self.Move(tc['last < >'], 'Q'), level)
                changed = True
        return changed

    def deduct_one_queen_per_zone(self, level: int):
        """
        Marks all empty cells in a zone with an 'X' if there is already a queen placed in that zone.

        This method iterates through each zone on the board. If a zone contains exactly one queen and at least 
        one empty cell, it marks all remaining empty cells in that zone as blocked ('X').

        :param level: The current recursive depth, used to undo moves if a dead end is reached.

        :return: True if any change was made in the board; otherwise, False.
        """
        changed = False
        for zone in self.zone_lists:
            tc = self.tiles_count(zone)
            if self.is_invalid:
                break
            if tc['Q'] == 1 and 0 < tc[' '] < len(self.zone_lists[zone]):
                for coord in self.zone_lists[zone]:
                    if self.board[coord] == ' ':
                        self.make_move(self.Move(coord, 'X'), level)
                        changed = True
        return changed

    def deduct_no_touching_queens(self, level: int):
        """
        Marks all cells surrounding a queen with an 'X' to enforce the rule that queens cannot touch each other, 
        not even diagonally.

        This method iterates through all queens on the board. For each queen, it marks all adjacent cells (up, down, 
        left, right, and diagonals) as blocked ('X'). If any adjacent cell already contains a queen, the board is 
        marked as invalid. Additionally, it checks if the solution is complete by verifying that there is exactly 
        one queen per row and no invalid state.

        :param level: The current recursive depth, used to undo moves if a dead end is reached.

        :return: True if any change was made in the board; otherwise, False.
        """
        changed = False
        queen_count = 0
        for zone in [z for z in self.zone_queens if z.startswith('row')]:
            self.tiles_count(zone)
            if self.is_invalid:
                break
            if self.zone_queens[zone] is None:
                continue
            queen_count += 1
            row, col = self.zone_queens[zone]
            for r, c in [(row + r, col + c) for r in (-1, 0, 1) for c in (-1, 0, 1)]:
                if r < 0 or r >= self.rows or c < 0 or c >= self.cols or (r == row and c == col):
                    continue
                if self.board[r, c] == ' ':
                    self.make_move(self.Move((r, c), 'X'), level)
                    changed = True
                elif self.board[r, c] == 'Q':
                    self.is_invalid = True
        self.is_solved = (queen_count == self.rows) and not self.is_invalid
        return changed

    def deduct_zone_qty_match_range_length(self, level: int):
        """
        This method checks if there are exactly `n` distinct zones within a contiguous range of `n` rows or `n` columns. 
        If this condition is met, it implies that there are `n` queens for these `n` rows (or columns) and exactly these 
        `n` zones. Consequently, any cells belonging to other zones within this range are marked as blocked ('X'). If the 
        number of distinct zones in the range exceeds `n`, the board is marked as invalid.

        :param level: The current recursive depth, used to undo moves if a dead end is reached.

        :return: True if any change was made in the board; otherwise, False.
        """
        changed = False
        zone_limits = {zone: self.zone_limits(zone) for zone in self.zone_lists if zone.startswith('zone')}
        zone_limits = {zone: zone_limits[zone] for zone in zone_limits if zone_limits[zone]}
        # check rows
        for range_length in range(1, self.rows + 1):
            for range_start in range(self.rows + 1 - range_length):
                range_end = range_start + range_length
                for c in range(2):  # for each coordinate, 0: rows, 1: columns
                    in_range_zones = {
                        z for z in zone_limits
                        if zone_limits[z]['min'][c] >= range_start and zone_limits[z]['max'][c] < range_end
                    }
                    if len(in_range_zones) == range_length:
                        for zone in set(zone_limits) - in_range_zones:
                            for coord in self.zone_lists[zone]:
                                if range_start <= coord[c] < range_end and self.board[coord] == ' ':
                                    self.make_move(self.Move(coord, 'X'), level)
                                    changed = True
                    elif len(in_range_zones) > range_length:
                        self.is_invalid = True
        return changed

    def deduct_zone_needed_within_range(self, level: int):
        """
        If part of a zone is needed to reach the number of queens in a range (rows or cols), then the zone cells outside
        the range must be marked X

        :param level: The current recursive depth, used to undo moves if a dead end is reached.

        :return: True if any change was made in the board; otherwise, False.
        """
        # TODO: If further optimization is needed to reduce brute-force searching, implement this and other possible deduction methods.
        changed = False

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
        while changed:
            changed = False
            changed = changed or self.deduct_no_touching_queens(level)
            if self.is_invalid or self.is_solved:
                break
            changed = changed or self.deduct_one_queen_per_zone(level)
            if self.is_invalid or self.is_solved:
                break
            changed = changed or self.deduct_one_place_left(level)
            if self.is_invalid or self.is_solved:
                break
            changed = changed or self.deduct_zone_qty_match_range_length(level)
            if self.is_invalid or self.is_solved:
                break
            changed = changed or self.deduct_zone_needed_within_range(level)
            if self.is_invalid or self.is_solved:
                break

    def clean_level(self, level: int):
        """
        Removes all moves made by make_move() and make_deduct_moves() at the specified recursive level.

        :param level: The recursive level used to undo all moves that led to a dead end.
        """
        self.moves_by_level[level].reverse()
        for m in self.moves_by_level[level]:
            self.board[m['coord']] = m['prev_tile']
        self.moves_by_level.pop(level)
        self.is_invalid = False

