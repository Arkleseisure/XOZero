from math import sqrt, log
from numpy.random import gamma
from numpy import array
from Bits_and_Pieces import make_dict


class Node:
    def __init__(self, prior, to_play):
        self.visit_count = 0
        self.value_sum = 0
        self.to_play = to_play
        self.prior = prior
        self.children = {}

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum/self.visit_count


def add_exploration_noise(config, node):
    actions = node.children.keys()
    noise = gamma(config.root_dirichlet_alpha, 1, len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


def select_child(config, node):
    _, action, child = max((ucb_score(config, node, child), action, child)
                           for action, child in node.children.items())
    return action, child


def ucb_score(config, parent, child):
    pb_c = log((parent.visit_count + config.pb_c_base + 1) / config.pb_c_base) + config.pb_c_init
    pb_c *= child.prior * sqrt(parent.visit_count) / (child.visit_count + 1)
    return pb_c + child.value()


def make_children(game, node, logits):
    move_dict = make_dict(logits, game.legal_moves)
    for move in move_dict:
        node.children[move] = Node(move_dict[move], 1 - node.to_play)
    return node.children


def backpropagate(value, search_path):
    last_move = search_path[-1].to_play
    for node in search_path:
        node.visit_count += 1
        if node.to_play == last_move:
            node.value_sum += value
        else:
            node.value_sum -= value


def select_action(root):
    visit_counts = [(child.visit_count, action) for action, child in root.children.items()]
    _, move = max(visit_counts)
    return move


def get_neural_net_move(game, neural_net):
    value, logits = neural_net.predict(array([game.make_image()]))
    move_dict = make_dict(logits[0], game.legal_moves)
    _, move = max([(value, key) for key, value in move_dict.items()])
    return move


def get_move(game, neural_net, config, root=None, noise=True, self_play=False):
    max_visits = config.num_simulations
    if not root:
        root = Node(0, game.to_play)
    value, logits = neural_net.predict(array([game.make_image()]))
    root.children = make_children(game, root, logits[0])
    if noise:
        add_exploration_noise(config, root)
    if game.num_moves == 8:
        return game.legal_moves[0], {game.legal_moves[0]: config.num_simulations}, root
    for i in range(config.num_simulations):
        node = root
        search_path = [root]
        exploration_game = game.clone()
        while node.expanded():
            move, node = select_child(config, node)
            exploration_game.apply(move)
            search_path.append(node)

        terminal = exploration_game.terminal()

        if not terminal:
            value, logits = neural_net.predict(array([exploration_game.make_image()]))
            value = value[0][0]
            node.children = make_children(exploration_game, node, logits[0])
        elif terminal == 'full':
            value = 0
        else:
            value = 1

        backpropagate(value, search_path)

        if not self_play:
            if i > config.num_simulations//2 and max_visits > config.num_simulations - i:
                visit_count_list = list(sorted(((key, root.children[key].visit_count) for key in root.children),
                                               key=lambda x: x[1]))
                max_visits = visit_count_list[-1][1]
                if max_visits - visit_count_list[-2][1] > config.num_simulations - i:
                    return visit_count_list[-1][0], \
                           {key: root.children[key].visit_count for key in root.children.keys()}, \
                           root.children[visit_count_list[-1][0]]

    move = select_action(root)
    return move, {key: root.children[key].visit_count for key in root.children.keys()}, root.children[move]
