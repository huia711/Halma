import pygame
from halma.constants import *


# 棋子类
class Piece:
    """
    A class representing a piece on the Halma board.

    Attributes:
        row (int): the row index of the piece on the board
        col (int): the column index of the piece on the board
        color (str): the color of the piece (either "BLACK" or "WHITE")
        x (int): the x-coordinate of the piece on the game window
        y (int): the y-coordinate of the piece on the game window
    """

    def __init__(self, row, col, color):
        """
        Initializes a Piece object with the given row, column, and color.

        Args:
            row (int): the row index of the piece on the board
            col (int): the column index of the piece on the board
            color (str): the color of the piece (either "BLACK" or "WHITE")
        """
        self.row = row
        self.col = col
        self.color = color
        self.x = 0
        self.y = 0
        self.calc_pos()

    # 以(row, col)返回当前位置
    def position(self):
        """
        Returns the current position of the piece as a tuple (row, col).

        Returns:
            A tuple representing the current position of the piece, with the row
            and column indices as the first and second elements of the tuple, respectively.
        """
        return self.row, self.col

    # 基于(row, col)计算(x, y)，以更新(x, y)
    def calc_pos(self):
        """
        Calculates the piece's x and y positions based on its row and column.
        """
        self.x = SQUARE_SIZE * self.col + SQUARE_SIZE // 2
        self.y = SQUARE_SIZE * self.row + SQUARE_SIZE // 2

    # 移动棋子
    def move(self, row, col):
        """
        Moves the piece to the specified row and column.

        Parameters:
            row (int): The row to move the piece to.
            col (int): The column to move the piece to.
        """
        self.row = row
        self.col = col
        self.calc_pos()

    # 绘制棋子
    def draw(self, win):
        """
         Draws the piece on the given window.

         Parameters:
             win (pygame.Surface): The window surface to draw on.
         """
        radius = SQUARE_SIZE//2 - PADDING

        # Draw outline
        pygame.draw.circle(win, GREY, (self.x, self.y), radius + OUTLINE)

        # Draw circle
        pygame.draw.circle(win, self.color, (self.x, self.y), radius)

    def __repr__(self):
        """
        Returns a string representation of the piece's color.

        Returns:
            str: The string representation of the piece's color.
        """
        return str(self.color)
