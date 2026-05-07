"""
Tic Tac Toe Player
"""

import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    x_count = sum(row.count(X) for row in board)
    o_count = sum(row.count(O) for row in board)

    return X if x_count == o_count else O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    return {
        (i, j)
        for i in range(3)
        for j in range(3)
        if board[i][j] == EMPTY
    }


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    if action not in actions(board):
        raise Exception("Invalid action")

    i, j = action
    new_board = [row.copy() for row in board]
    new_board[i][j] = player(board)

    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    lines = []
    lines.extend(board)
    lines.extend([[board[0][j], board[1][j], board[2][j]] for j in range(3)])
    lines.append([board[0][0], board[1][1], board[2][2]])
    lines.append([board[0][2], board[1][1], board[2][0]])

    for line in lines:
        if line[0] is not EMPTY and line[0] == line[1] == line[2]:
            return line[0]

    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    return winner(board) is not None or not actions(board)


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    winning_player = winner(board)

    if winning_player == X:
        return 1
    if winning_player == O:
        return -1
    return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None

    turn = player(board)
    best_action = None

    if turn == X:
        best_value = -math.inf
        for action in actions(board):
            value = min_value(result(board, action))
            if value > best_value:
                best_value = value
                best_action = action
    else:
        best_value = math.inf
        for action in actions(board):
            value = max_value(result(board, action))
            if value < best_value:
                best_value = value
                best_action = action

    return best_action


def max_value(board):
    """
    Returns the best utility X can force from a board.
    """
    if terminal(board):
        return utility(board)

    value = -math.inf
    for action in actions(board):
        value = max(value, min_value(result(board, action)))

    return value


def min_value(board):
    """
    Returns the best utility O can force from a board.
    """
    if terminal(board):
        return utility(board)

    value = math.inf
    for action in actions(board):
        value = min(value, max_value(result(board, action)))

    return value
