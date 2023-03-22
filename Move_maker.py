from random import randint, shuffle
from copy import deepcopy
from Bits_and_Pieces import draw_board, make_dict_key
from Game import Game


def get_random_move(game):
    x = randint(0, 2)
    y = randint(0, 2)
    while not game.board[0][x][y] == 0 == game.board[1][x][y]:
        x = randint(0, 2)
        y = randint(0, 2)
    return str(x) + str(y)


def minimax(state_dict, value_dict, game=None):
    if not game:
        game = Game()
    value = 3 * game.to_play - 1
    shuffle(game.legal_moves)
    possible_moves = deepcopy(game.legal_moves)
    key = make_dict_key(game.board)
    for move in possible_moves:
        game.apply(move)
        result = game.terminal()
        if result:
            if result == 'full':
                node_value = 0.5
            else:
                node_value = game.to_play
        else:
            node_value = minimax(state_dict, value_dict, game)
        game.to_play = 1 - game.to_play
        game.board[game.to_play][int(move[0])][int(move[1])] = 0
        game.num_moves -= 1
        game.legal_moves.append(move)
        if (node_value > value and game.to_play == 0) or (node_value < value and game.to_play == 1):
            value = node_value
            state_dict[key] = [move]
            value_dict[key] = value
        elif node_value == value:
            if key in state_dict.keys():
                state_dict[key].append(move)
            else:
                state_dict[key] = [move]
                value_dict[key] = value
    return value


def get_player_move(game):
    draw_board(game.board)
    x = int(input('Please enter the x coordinate of your move: ')) - 1
    y = int(input('Please enter the y coordinate of your move: ')) - 1
    while not game.board[0][x][y] == 0 == game.board[1][x][y]:
        print('That is not a correct input')
        x = int(input('Please enter the x coordinate of your move: ')) - 1
        y = int(input('Please enter the y coordinate of your move: ')) - 1
    return str(x) + str(y)
