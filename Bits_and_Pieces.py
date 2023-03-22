from math import exp


def make_dict_key(board):
    key = ''
    for player in board:
        for row in board[player]:
            for item in board[player][row].values():
                key += str(item)
    return key


def to_time(secs):
    return str(round(secs//3600)) + ":" + str(round((secs % 3600) // 60)) + ":" + str(secs % 60)


def draw_board(board):
    for i in range(len(board[0])):
        for j in range(len(board[0][i])):
            item = ' '
            if board[0][j][i] == 1 == board[1][j][i]:
                item = 'B'
            elif board[0][j][i] == 1:
                item = 'X'
            elif board[1][j][i] == 1:
                item = 'O'
            print(item, end='|' if j != 2 else '\n')
    print()


def make_dict(pred, legal_moves):
    move_dict = {}
    total = 0
    for move in legal_moves:
        move_dict[move] = exp(pred[int(move[0]) + int(move[1]) * 3])
        total += move_dict[move]

    for key in move_dict:
        move_dict[key] = move_dict[key]/total

    return move_dict
