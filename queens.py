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
        self.zone_matrix = None
        self.board = None
        self.rows, self.cols = 0, 0
        self.moves_by_level = dict()
        self.zone_lists = None
        self.zone_queens = dict()
        self.order_within_level = 0
        self.is_solved = False
        self.is_invalid = False
        self.image = None
        self.x_axis = None
        self.y_axis = None
        self.size = None
        self.zone_colors = None
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
        Checks if the puzzle has been solved.

        :return: True if the puzzle is solved, otherwise returns False.
        """
        return self.is_solved and self.board is not None

    def is_dead_end_or_invalid(self):
        """
        Determines if the current board state is invalid or if continuing from this position is futile.

        :return: True if the board is in an invalid state or if the current situation clearly leads to a dead end.
                 Otherwise, returns False.
        """
        return self.is_invalid and self.board is not None

    def candidate_moves(self, level: int):
        """
        Generates an iterable of possible moves to make from the current board position.
        The moves are ordered to increase the likelihood of finding a solution quickly.

        :param level: Indicates the current recursive depth of the search, useful for undoing all moves
                      if a dead end is reached.
        :return: An iterable containing all the possible moves from the current position.
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
        Executes a move on the board and stores the metadata needed to undo this move
        and others made at the same recursive level.

        :param move: An object containing the move details, such as coordinates, type of piece, character, etc.
        :param level: Indicates the current recursive depth of the search, useful for undoing all moves
                      if a dead end is reached.
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
        Counts tiles of each type in the specified zone
        Marks invalid the puzzle if the zone has more than one queen or has no room for a queen (all 'X's)
        Stores the position of the last queen encountered in the zone (if any)
        :param zone: the zone to count
        :return: a dict containing counters for the three types of tile and the last position of each one
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
        If there is just one place left in a row, column o zone, put a queen there
        :return: True if any change was made in the board
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
        If there is a queen in a zone, mark with X the other cells of that zone
        :return: True if any change was made in the board
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
        If there is a queen in a cell, mark with X all the cells around it
        :return: True if any change was made in the board
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
        If there is exactly n zones enclosed in an n rows (or cols) range, then the other zones within the range must be
        marked X, because we have n queens for n rows (or cols) and these exactly n zones.
        :return: True if any change was made in the board
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
        :return: True if any change was made in the board
        """
        changed = False

        return changed

    def make_deduct_moves(self, level: int):
        """
        Executes all mandatory moves based on the current board state. These moves are labeled with the same
        recursive level, allowing them to be undone if this position, along with the mandatory moves, leads
        to an invalid state or dead end.

        If a dead end is reached, the clean_level() method will remove all moves marked with the current level,
        including the move that led to this position.

        :param level: Indicates the current recursive depth of the search, useful for undoing all moves
                      if a dead end is encountered.
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
        Removes all moves made by make_move() and make_deduct_moves() that are labeled with the specified level.

        :param level: Indicates the recursive level of the search, used to undo all moves that led to a dead end.
        """
        self.moves_by_level[level].reverse()
        for m in self.moves_by_level[level]:
            self.board[m['coord']] = m['prev_tile']
        self.moves_by_level.pop(level)
        self.is_invalid = False

