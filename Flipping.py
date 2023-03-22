def generate_flipped_data(input_data, target_value_data, target_policy_data):
    new_input_data = []
    new_value_data = []
    new_policy_data = []
    for i in range(len(input_data)):
        for j in range(7):
            new_input_data.append(flip(input_data[i], j))
            new_policy_data.append(flip_result(target_policy_data[i], j))
            new_value_data.append(target_value_data[i])
    input_data.extend(new_input_data)
    target_value_data.extend(new_value_data)
    target_policy_data.extend(new_policy_data)


def flip(board, num):
    new_board = [[[0 for i in range(len(board[0]))] for j in range(len(board[0]))] for k in range(2)]
    new_board.append(board[2])
    for i in range(2):
        for j in range(len(board[0])):
            for k in range(len(board[0])):
                if num == 0:  # flip in the x direction
                    new_board[i][2 - j][k] = board[i][j][k]
                elif num == 1:  # flip in the y direction
                    new_board[i][j][2 - k] = board[i][j][k]
                elif num == 2:  # flip across 02 20 axis
                    new_board[i][2 - k][2 - j] = board[i][j][k]
                elif num == 3:  # flip across 00 22 axis
                    new_board[i][k][j] = board[i][j][k]
                elif num == 4:  # 90 degree anticlockwise rotation
                    new_board[i][k][2 - j] = board[i][j][k]
                elif num == 5:  # 90 degree clockwise rotation
                    new_board[i][2 - k][j] = board[i][j][k]
                elif num == 6:  # 180 degree rotation
                    new_board[i][2 - j][2 - k] = board[i][j][k]
    return new_board


def flip_result(result, num):
    new_result = [0 for i in range(len(result))]
    for i in range(len(result)):
        if num == 0:
            # loc = i - x + 2 - x, where x = i % 3
            new_result[2 * (1 - i % 3) + i] = result[i]
        elif num == 1:
            # loc = i - 3 * y + 3 * (2 - y), where y = i // 3
            new_result[6 * (1 - i // 3) + i] = result[i]
        elif num == 2:
            # loc = 2 - y + 3 * (2 - x)
            new_result[8 - i// 3 - 3 * (i % 3)] = result[i]
        elif num == 3:
            # loc = 3 * x + y
            new_result[3 * (i % 3) + i // 3] = result[i]
        elif num == 4:
            # loc = y + 3 * (2 - x)
            new_result[6 + i // 3 - 3 * (i % 3)] = result[i]
        elif num == 5:
            # loc = 2 - y + 3 * x
            new_result[2 - i // 3 + 3 * (i % 3)] = result[i]
        elif num == 6:
            # loc = 3 * (2 - y) + (2 - x)
            new_result[8 - 3 * (i // 3) - i % 3] = result[i]
    return new_result
