from time import time
from random import randint
from Bits_and_Pieces import make_dict_key
from Net_move_maker import get_move, get_neural_net_move, add_exploration_noise, Node, make_children
from Move_maker import get_random_move, get_player_move
from Game import Game
from numpy import array


# mode takes two letters as input, representing the two players, the one starting being the first letter.
# self-play = self-play (only option not made up of two letters)
# s = neural network with search
# n = neural network
# r = random
# m = minimax (perfect)
# h = human
def play_game(neural_net, config, mode='hh', state_dict={}, neural_net_2=None, noise=False):
    game = Game()
    move_time = 0
    moves = 0
    while not game.terminal():
        if mode[game.to_play] == 's':
            start_time = time()
            move, search_stats, next_root = get_move(game, neural_net, config, noise=noise)
            move_time += (time() - start_time)
            moves += 1
        elif mode[game.to_play] == 'n':
            if noise:
                root = Node(0, game.to_play)
                value, logits = neural_net.predict(array([game.make_image()]))
                root.children = make_children(game, root, logits[0])
                add_exploration_noise(config, root)
                _, move = max((root.children[key].prior, key) for key in root.children)
            else:
                move = get_neural_net_move(game, neural_net)
        elif mode[game.to_play] == 'S':
            move, search_stats, next_root = get_move(game, neural_net_2, config, noise=noise)
        elif mode[game.to_play] == 'N':
            if noise:
                root = Node(0, game.to_play)
                value, logits = neural_net_2.predict(array([game.make_image()]))
                root.children = make_children(game, root, logits[0])
                add_exploration_noise(config, root)
                _, move = max((root.children[key].prior, key) for key in root.children)
            else:
                move = get_neural_net_move(game, neural_net_2)
        elif mode[game.to_play] == 'r':
            move = get_random_move(game)
        elif mode[game.to_play] == 'h':
            move = get_player_move(game)
        elif mode[game.to_play] == 'm':
            board_key = make_dict_key(game.board)
            move = state_dict[board_key][randint(0, len(state_dict[board_key]) - 1)]
        try:
            game.apply(move)
        except ValueError:
            print(move)
            print(game.legal_moves)
    result = game.terminal()
    if 'r' in mode or 'm' in mode or 'N' in mode or 'S' in mode:
        return 0.5 if result == 'full' else game.to_play, move_time, moves
