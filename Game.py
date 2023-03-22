from copy import deepcopy, copy


board_size = 3


class Game:
    def __init__(self, board=None, to_play=None, last_move=None, legal_moves=None, num_moves=None):
        if board:
            self.board = deepcopy(board)
            self.to_play = to_play
            self.last_move = last_move
            self.legal_moves = copy(legal_moves)
            self.num_moves = num_moves
        else:
            self.legal_moves = [str(i) + str(j) for i in range(board_size) for j in range(board_size)]
            self.board = {k: {j: {i: 0 for i in range(board_size)} for j in range(board_size)} for k in range(2)}
            self.to_play = 0
            self.last_move = '00'
            self.num_moves = 0

    def apply(self, move):
        self.board[self.to_play][int(move[0])][int(move[1])] = 1
        self.to_play = 1 - self.to_play
        self.last_move = move
        self.legal_moves.remove(move)
        self.num_moves += 1

    def terminal(self):
        vertical = True
        horizontal = True
        diag_1 = True
        diag_2 = True
        for i in range(board_size):
            if self.board[1 - self.to_play][int(self.last_move[0])][i] != 1:
                vertical = False
            if self.board[1 - self.to_play][i][int(self.last_move[1])] != 1:
                horizontal = False
            if self.board[1 - self.to_play][i][i] != 1:
                diag_1 = False
            if self.board[1 - self.to_play][i][2 - i] != 1:
                diag_2 = False

            if not (vertical or horizontal or diag_1 or diag_2):
                if self.num_moves == 9:
                    return 'full'
                return False
        return True

    def make_image(self):
        return [[[self.board[self.to_play][key][key2] for key2 in self.board[self.to_play][key]]
                 for key in self.board[self.to_play]],
                [[self.board[1 - self.to_play][key][key2] for key2 in self.board[1 - self.to_play][key]]
                 for key in self.board[1 - self.to_play]],
                [[self.to_play for key2 in self.board[0][key]] for key in self.board[0]]]

    def clone(self):
        return Game(self.board, self.to_play, self.last_move, self.legal_moves, self.num_moves)
