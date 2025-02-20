import pygame
from halma.board import Board
from halma.constants import *


# 游戏总类
class Game:
    """
    The Game class represents the main game logic of the halma game.
    It initializes the game board and the game window, and handles user input
    for selecting and moving pieces on the board. It also determines the winner
    of the game and resets the game state.
    """

    def __init__(self, win):
        # 初始化游戏变量
        self._init()
        # 绘制游戏窗口
        self.win = win

    # 更新棋盘状态
    def update(self):
        self.board.draw(self.win)  # 更新棋盘
        self.draw_valid_moves(self.valid_moves)  # 标识出有效走法
        self.draw_selected(self.selected)  # 标识出已经选择棋子
        pygame.display.update()

    # 初始化游戏变量
    def _init(self):
        self.selected = None  # 选择的棋子
        self.board = Board()  # 棋盘
        self.turn = BLACK  # 回合，黑先手
        self.valid_moves = []  # 有效移动

    # 返回棋盘状态（二维列表）
    def get_board(self):
        return self.board

    # 将游戏状态重置为初始条件
    def reset(self):
        self._init()

    # 处理棋子的选择和移动
    def select(self, row, col):
        # 如果已经选择棋子【移动】
        if self.selected:
            # 试着将棋子移动到新的位置
            result = self._move(row, col)
            # 如果移动无效
            if not result:
                # 将棋子选择释放
                self.selected = None
                self.select(row, col)

        # 如果没有选棋子【选棋】
        else:
            # 得到该位置棋子
            piece = self.board.get_piece(row, col)
            # 如果该位置是正确的棋子
            if piece != 0 and piece.color == self.turn:
                self.selected = piece
                self.valid_moves = self.board.get_valid_moves(piece)  # 返回棋子的有效走法列表

                return True

        return False

    # 将棋子移动到新的位置
    def _move(self, row, col):
        # 得到该位置棋子
        piece = self.board.get_piece(row, col)

        # 如果新的位置可以移动
        if self.selected and piece == 0 and (row, col) in self.valid_moves:
            # 移动棋子
            self.board.move(self.selected, row, col)
            # 下一回合
            self.change_turn()

        else:
            return False

        return True

    # 标识出已经选择棋子【绿圈】
    def draw_selected(self, selected):
        if self.selected:
            pygame.draw.circle(self.win, LIGHT_GREEN, (selected.x, selected.y), 47, 10)

    # 标识出棋子有效走法【蓝圈】
    def draw_valid_moves(self, moves):
        # 遍历有效走法列表
        for move in moves:
            row, col = move

            # the circle is centered at the middle of the square and has a radius of 15 pixels
            pygame.draw.circle(
                self.win,
                EMERALD,
                (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2),
                15
            )

    # 转换回合
    def change_turn(self):
        # 清除可走列表
        self.valid_moves = []
        # 清除已选择棋子
        self.selected = None
        # 转换回合
        if self.turn == BLACK:
            self.turn = WHITE
        else:
            self.turn = BLACK

    # 决定这场比赛的获胜者
    def winner(self):
        return self.board.winner()
