from copy import deepcopy
import pygame.draw

# 颜色
BLACK = (0, 0, 0)  # user piece color
WHITE = (255, 255, 255)  # AI piece color


# minimax函数：实现极小化极大算法与Alpha-Beta剪枝
def minimax(board, depth, alpha, beta, max_player, game=None):
    """
    This function uses the minimax algorithm with alpha-beta pruning
    to search for the best move for the current player.

    Parameters:
        board (Board): The current state of the board.
        depth (int): The current depth of the search.
        alpha (int): The current alpha value for alpha-beta pruning.
        beta (int): The current beta value for alpha-beta pruning.
        max_player (bool): True if the current player is the maximizing player, False otherwise.
        game (Game): The current game instance.

    Returns:
        (int, Board): A tuple containing the evaluation score and the board state of the best move found.
    """

    # 检查是否达到了最大搜索深度，或者是否找到了赢家
    if depth == 0 or board.winner() is not None:
        # 返回棋盘评估值、棋盘状态
        return board.evaluate(), board

    # 最大化玩家【evaluate大】
    if max_player:
        # 最大玩家试图最大化评估分数
        max_evaluation = float("-inf")
        best_move = None

        # 遍历所有有效移动，评估每个移动
        for move in get_all_moves(board, WHITE, game):
            # 递归下一回合【最小化玩家】
            evaluation = minimax(move, depth-1, alpha, beta, False, game)[0]

            # 更新最大评价分数和最佳走法
            if evaluation > max_evaluation:
                max_evaluation = evaluation
                best_move = move
            # 更新alpha值
            alpha = max(alpha, evaluation)

            # 如果beta小于等于alpha，跳出循环
            if beta <= alpha:
                break

        return max_evaluation, best_move

    # 最小化玩家【evaluate小】
    else:
        # 最小玩家试图最小化评估分数
        min_evaluation = float("inf")
        best_move = None

        # 遍历所有有效移动，评估每个移动
        for move in get_all_moves(board, BLACK, game):
            # 递归下一回合【最大化玩家】
            evaluation = minimax(move, depth - 1, alpha, beta, True, game)[0]

            # 更新最小评价分数和最佳走法
            if evaluation < min_evaluation:
                min_evaluation = evaluation
                best_move = move
            # 更新beta值
            beta = min(beta, evaluation)

            # 如果beta小于等于alpha，跳出循环
            if beta <= alpha:
                break
        return min_evaluation, best_move


# 返回棋盘上给定颜色的所有可能走法
def get_all_moves(board, color, game=None):
    """
    Returns all possible moves for a given color on the current board.

    Parameters:
        board (Board): current game board
        color (string): color of the pieces to move (either RED or WHITE)
        game (Game): current game object

    Returns:
        list: a list of Board objects representing all possible board
        configurations after a move by the given color
    """
    moves = []

    # 遍历给定颜色的所有棋子
    for piece in board.get_all_pieces(color):
        # 获得当前棋子的所有有效走法
        valid_moves = board.get_valid_moves(piece)

        # 遍历当前棋子的所有有效走法
        for move in valid_moves:
            # 可视化决策过程 (optional)
            # visualize(game, board, piece)

            # 创建一个副本
            temp_board = deepcopy(board)
            temp_piece = temp_board.get_piece(piece.row, piece.col)
            # 模拟移动
            new_board = simulate_move(temp_piece, move, temp_board)
            # 将结果添加到moves
            moves.append(new_board)

    return moves


# 模拟移动
def simulate_move(piece, move, board):
    """
    Simulates a given move on a copy of the current board.

    Args:
        piece (Piece): piece to move
        move (tuple): new position of the piece (row, col)
        board (Board): current game board
        game (Game): current game object

    Returns:
        Board: a new board configuration after the given move has been made
    """
    # 将棋子移动到棋盘上的新位置
    board.move(piece, move[0], move[1])

    return board


# 可视化决策过程 (optional)
def visualize(game, board, piece):
    """
    Visualizes the minimax algorithm by drawing the current board state and highlighting the valid moves for a piece.

    Parameters:
        game (Game): the current Game object.
        board (Board): the current Board object.
        piece (Piece): the Piece object to visualize.

    Returns:
        None
    """
    # Get valid moves for the piece
    valid_moves = board.get_valid_moves(piece)

    # Draw the board and highlight the selected piece
    board.draw(game.win)
    pygame.draw.circle(game.win, (93, 187, 99), (piece.x, piece.y), 47, 10)

    # Draw circles on valid move squares
    game.draw_valid_moves(valid_moves)

    # Update the display
    pygame.display.update()

    # Wait for a short delay (optional)
    pygame.time.delay(100)
