import abc
from datetime import datetime, timedelta


class BoardPuzzle(abc.ABC):

    @abc.abstractmethod
    def solved(self):
        """
        Checks if the puzzle is solved.

        :return: True if the puzzle is solved; otherwise, False.
        """
        pass

    @abc.abstractmethod
    def is_dead_end_or_invalid(self):
        """
        Determines whether the current board state is invalid or leads to a dead end.

        :return: True if the board state is invalid or leads to a dead end; otherwise, False.
        """
        pass

    @abc.abstractmethod
    def candidate_moves(self, level: int):
        """
        Generates an iterable of possible moves from the current board state, ordered to maximize the chances 
        of finding a solution quickly.

        :param level: The current recursive depth, useful for undoing all moves if a dead end is reached.
        :return: An iterable of possible moves from the current position.
        """
        pass

    @abc.abstractmethod
    def make_move(self, move, level: int):
        """
        Executes a move and records metadata needed to undo this and other moves at the same recursive level.

        :param move: An object representing the move, such as coordinates, piece type, or character.
        :param level: The current recursive depth, used to undo moves if a dead end is reached.
        """
        pass

    @abc.abstractmethod
    def make_deduct_moves(self, level: int):
        """
        Executes all mandatory moves based on the current board state. These moves are labeled with the 
        current recursive level, allowing them to be undone if they lead to an invalid state or dead end.

        If a dead end is reached, clean_level() will remove all moves marked with the current level.

        :param level: The current recursive depth, used to undo moves if a dead end is encountered.
        """
        pass

    @abc.abstractmethod
    def clean_level(self, level: int):
        """
        Removes all moves made by make_move() and make_deduct_moves() at the specified recursive level.

        :param level: The recursive level used to undo all moves that led to a dead end.
        """
        pass


class BPSolver:

    def __init__(self, puzzle: BoardPuzzle, max_moves: int, max_time: timedelta, max_level: int):
        """
        Board Puzzle Solver

        Attempts to solve a board puzzle by exploring possible moves and making logical deductions. A board puzzle typically involves placing objects while adhering to specific rules or constraints. Examples include arranging chess pieces on a board or filling a grid with numbers, tiles, or characters under certain conditions.

        This class takes an instance of a class that implements the abstract `BoardPuzzle` interface and searches for a solution. You only need to provide the puzzle object.

        The solving strategy follows these steps:
            1) Whenever possible, deduce and apply any mandatory moves based on the current board state.
            2) When no further deductions can be made, make an educated guess—selecting the move most likely to lead to a solution—and then return to step 1.
            3) If the chosen path reaches a dead end, backtrack to the previous board state and make a new guess.

        The efficiency of the solver largely depends on how well the `BoardPuzzle` methods are implemented. For instance:
            - Detecting dead-end paths early can significantly reduce the search space.
            - Making more deductions to identify mandatory moves reduces the need for guesses.
            - Optimizing the order in which potential moves are evaluated can improve search effectiveness.

        :param puzzle: an instance of a class that implements the abstract `BoardPuzzle` interface
        :param max_moves: the maximum number of moves the solver will attempt before giving up, leaving the puzzle unsolved
        :param max_time: the maximum time allowed for solving the puzzle; if exceeded, the solver will time out and leave the puzzle unsolved
        :param max_level: the maximum number of recursive calls allowed while attempting to solve the puzzle
        """
        self.puzzle = puzzle
        self.max_moves = max_moves
        self.max_time = max_time
        self.max_level = max_level
        self.solved = False
        self.invalid_path = False
        self.move_count = 0
        self.start = datetime.now()
        self.level_watermark = 0

    def quit(self):
        """
        Returns the reason for stopping, if any; otherwise, returns an empty string.
        """
        if (datetime.now() - self.start) > self.max_time:
            return 'TIME OUT'
        if self.move_count > self.max_moves:
            return 'MAX MOVE COUNT REACHED'
        if self.level_watermark > self.max_level:
            return 'MAXIMUM RECURSION LEVEL REACHED'
        return ''

    def solve(self, level=0):
        """
        Attempts to solve the puzzle. Refer to the class constructor for detailed information on the solving strategy.

        This method recursively searches for a solution by making deductions and testing possible moves. If a dead end 
        or invalid path is encountered, it backtracks by undoing all moves made at the specified recursion depth. 
        The `level` parameter serves both to track recursion depth and to label moves for cleanup during backtracking.

        :param level: The current recursion depth, used to prevent infinite loops, log debugging or additional information 
                        within the puzzle object, and identify which moves to clean up during backtracking.
        """
        self.level_watermark = max(level, self.level_watermark)
        self.puzzle.make_deduct_moves(level)
        self.solved = self.puzzle.solved()
        self.invalid_path = self.puzzle.is_dead_end_or_invalid()
        if self.solved or self.invalid_path:
            return 'OK' if self.solved else 'COULD NOT FIND ANY SOLUTION'

        quit_reason = ''
        for move in self.puzzle.candidate_moves(level):
            self.move_count += 1
            quit_reason = self.quit()
            if quit_reason:
                break
            self.puzzle.make_move(move, level + 1)
            self.solve(level + 1)
            if self.solved:
                break
            else:
                self.puzzle.clean_level(level + 1)

        return 'OK' if self.solved else quit_reason or 'COULD NOT FIND ANY SOLUTION'
