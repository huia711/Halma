import time
import pygame
from halma.constants import *
from halma.game import Game
from minimax.algorithm import minimax, simulate_move
from ppo import ppo_move
import argparse


# 窗口
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Halma")  # 标题
FPS = 60  # 帧率


# 将位置转为棋盘行列
def get_row_col_from_mouse(pos):
    x, y = pos
    row = y // SQUARE_SIZE
    col = x // SQUARE_SIZE
    return row, col


# # 解析命令行参数
# def set_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--play', action='store_false', default=True)  # 如果用户在命令行中指定了’–play’参数，则’play’变量被设置为False
#     parser.add_argument('--test', action='store_true', default=False)  # 如果用户在命令行中指定了’–test’参数，则’test’变量被设置为True
#
#     return parser.parse_args()  # 返回参数元组


def main():
    chance = 5
    run = True  # 指示游戏运行
    clock = pygame.time.Clock()  # Clock对象：控制游戏中的帧率
    game = Game(WIN)  # 游戏实例

    # 主进程
    while run:
        # 保持游戏运行时每秒钟不超过FPS帧
        clock.tick(FPS)
        # 输出棋手的胜率
        print(game.get_board().evaluate())  # evaluate()：评估棋手的胜率

        # 轮到白棋
        if game.turn == WHITE:
            # 调用minimax算法，选择最佳移动
            value, new_board = minimax(game.get_board(), 1, float('-inf'), float('inf'), True, game)
            game.board = new_board  # 更新棋盘
            game.change_turn()  # 切换到对方的回合

            # # 监听鼠标点击事件
            # for event in pygame.event.get():
            #     # 是否鼠标点击事件
            #     if event.type == pygame.MOUSEBUTTONDOWN:
            #         pos = pygame.mouse.get_pos()  # 获得鼠标位置
            #         row, col = get_row_col_from_mouse(pos)  # 转为棋盘的行列
            #         game.select(row, col)  # 下子（判断是否空白格）

        # 轮到黑棋
        elif game.turn == BLACK:
            new_board = game.get_board()
            # 调用PPO算法，选择最佳移动
            if chance > 0:
                new_board = ppo_move(game.get_board(), game)
                chance = chance - 1
            elif chance == 0:
                value, new_board = minimax(game.get_board(), 1, float('-inf'), float('inf'), False, game)
                if value >= game.get_board().evaluate() or value > ppo_move(game.get_board(), game).evaluate():
                    new_board = ppo_move(game.get_board(), game)
                    chance = 1
            game.board = new_board  # 更新棋盘
            game.change_turn()  # 切换到对方的回合

        # 出现胜者
        if game.winner() is not None:
            print(game.winner())
            # 关闭游戏
            game.turn = None

        # 监听Pygame事件
        for event in pygame.event.get():
            # 是否点击窗口关闭按钮
            if event.type == pygame.QUIT:
                # 关闭游戏
                run = False

        # 更新显示效果
        game.update()
        time.sleep(0.1)  # 落子后在屏幕上停留1秒，以便观察游戏的进程

    pygame.quit()


if __name__ == '__main__':
    main()
