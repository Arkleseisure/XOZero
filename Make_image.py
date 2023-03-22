def make_output_image(search_stats, board_size):
    sum_visits = sum(stat for stat in search_stats.values())
    output = []
    for i in range(board_size ** 2):
        key = str(i % board_size) + str(i // board_size)
        if key in search_stats.keys():
            output.append(search_stats[key]/sum_visits)
        else:
            output.append(0)
    return output


def make_mm_output_image(game, state_dict):
    possible_moves = state_dict[make_dict_key(game.board)]
    output = []
    board_size = len(game.board[0])
    for i in range(board_size ** 2):
        key = str(i % board_size) + str(i // board_size)
        if key in possible_moves:
            output.append(1/len(possible_moves))
        else:
            output.append(0)
    return output
