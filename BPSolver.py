import abc
from datetime import datetime, timedelta


class BoardPuzzle(abc.ABC):

    @abc.abstractmethod
    def solved(self):
        """
        Checks if the puzzle has been solved.

        :return: True if the puzzle is solved, otherwise returns False.
        """
        pass

    @abc.abstractmethod
    def is_dead_end_or_invalid(self):
        """
        Determines if the current board state is invalid or if continuing from this position is futile.

        :return: True if the board is in an invalid state or if the current situation clearly leads to a dead end.
                 Otherwise, returns False.
        """
        pass

    @abc.abstractmethod
    def candidate_moves(self, level: int):
        """
        Generates an iterable of possible moves to make from the current board position.
        The moves are ordered to increase the likelihood of finding a solution quickly.

        :param level: Indicates the current recursive depth of the search, useful for undoing all moves
                      if a dead end is reached.
        :return: An iterable containing all the possible moves from the current position.
        """
        pass

    @abc.abstractmethod
    def make_move(self, move, level: int):
        """
        Executes a move on the board and stores the metadata needed to undo this move
        and others made at the same recursive level.

        :param move: An object containing the move details, such as coordinates, type of piece, character, etc.
        :param level: Indicates the current recursive depth of the search, useful for undoing all moves
                      if a dead end is reached.
        """
        pass

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
    def clean_level(self, level: int):
        """
        Removes all moves made by make_move() and make_deduct_moves() that are labeled with the specified level.

        :param level: Indicates the recursive level of the search, used to undo all moves that led to a dead end.
        """
        pass


class BPSolver:
    """
    Board Puzzle Solver

    Finds a solution to a board puzzle by testing possible moves and making deductions. A board puzzle requires placing objects
    while meeting certain rules or restrictions. For example, a chess board where you need to arrange certain pieces,
    or a grid where you need to put numbers, tiles, characters, etc., with some restrictions.

    This class takes an object of a class that implements the abstract class BoardPuzzle and searches for a solution.
    So all you need to provide is the board object.

    The strategy is:
        1) When it's possible to deduct some mandatory moves from the current board position, make them.
        2) When you cannot repeat step 1 anymore, make a guess and try your best move—the one most likely to lead to a
        solution quickly. Then go to step 1 from here.
        3) If that leads you to a dead end, return to the previous position and make a new guess.

    The more efficiently the puzzle object implements its methods, the higher the probability of finding a solution.
    For example, if you can detect dead-end paths early, you can dramatically cut the search tree. The more deductions
    you can make to generate mandatory moves, the less you'll need to make guesses. You can also control the order of
    the search by controlling the order in which the guess moves are tried from a given board position.
    """

    def __init__(self, puzzle: BoardPuzzle, max_moves: int, max_time: timedelta, max_level: int):
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
        if (datetime.now() - self.start) > self.max_time:
            return 'TIMEOUT'
        if self.move_count > self.max_moves:
            return 'MAX MOVE COUNT REACHED'
        if self.level_watermark > self.max_level:
            return 'MAXIMUM RECURSION LEVEL REACHED'
        return ''

    def solve(self, level=0):
        self.level_watermark = max(level, self.level_watermark)
        self.puzzle.make_deduct_moves(level)
        self.solved = self.puzzle.solved()
        self.invalid_path = self.puzzle.is_dead_end_or_invalid()
        if self.solved or self.invalid_path:
            return 'OK' if self.solved else 'COULD NOT FIND ANY SOLUTION'

        quit = ''
        for move in self.puzzle.candidate_moves(level):
            self.move_count += 1
            quit = self.quit()
            if quit:
                break
            self.puzzle.make_move(move, level+1)
            self.solve(level+1)
            if self.solved:
                break
            else:
                self.puzzle.clean_level(level+1)

        if self.solved:
            return 'OK'
        else:
            return quit if quit else 'COULD NOT FIND ANY SOLUTION'
