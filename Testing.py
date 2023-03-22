from time import time, ctime
from Play_game import play_game
if __name__ == 'Testing':
    from numpy import array, std, average, round
    from Game import Game
    from Bits_and_Pieces import draw_board, make_dict_key
    from Move_maker import get_random_move, minimax


def initializing_func_2(config):
    # stuff to stop it from printing a load of useless stuff to the screen
    from os import devnull
    from sys import stderr
    old_stderr = stderr
    stderr = open(devnull, 'w')
    from keras.models import load_model
    from tensorflow.compat.v1.keras.backend import set_session
    from tensorflow.compat.v1 import Session, ConfigProto, GPUOptions
    from tensorflow import nn
    stderr = old_stderr

    global m_config
    m_config = config
    tf_config = ConfigProto(gpu_options=GPUOptions(per_process_gpu_memory_fraction=m_config.max_mem / m_config.processes,
                                                   allow_growth=True))
    session = Session(config=tf_config)
    set_session(session)

    global m_neural_net
    m_neural_net = load_model(m_config.net_name, custom_objects={'softmax_cross_entropy_with_logits_v2':
                                                                 nn.softmax_cross_entropy_with_logits})


def print_visit_stats(game, node):
    draw_board(game.board)
    for child in node.children:
        print(child, node.children[child].visit_count, node.children[child].prior, node.children[child].value_sum,
              node.children[child].value())


def test_network(neural_net, config):
    state_dict = {}
    value_dict = {}
    minimax(state_dict, value_dict)
    print('Testing')
    score_r = 0
    score_m = 0
    move_time = 0
    moves = 0
    raw_net_score_r = 0
    raw_net_score_m = 0
    search_time = 0
    net_time = 0
    for j in range(config.checkpoint_game_num // 2):
        if (j + 1) % 5 == 0:
            print('Set:', (j + 1) * 2)
        try:
            start_time = time()
            a, extra_time, extra_moves = play_game(neural_net, config, mode='sr')
            move_time += extra_time
            moves += extra_moves
            b, extra_time, extra_moves = play_game(neural_net, config, mode='rs')
            move_time += extra_time
            moves += extra_moves
            score_r += 1 + a - b
            a, extra_time, extra_moves = play_game(neural_net, config, mode='sm', state_dict=state_dict)
            move_time += extra_time
            moves += extra_moves
            b, extra_time, extra_moves = play_game(neural_net, config, mode='ms', state_dict=state_dict)
            move_time += extra_time
            moves += extra_moves
            score_m += 1 + a - b
            search_time += time() - start_time

            start_time = time()
            a, extra_time, extra_moves = play_game(neural_net, config, mode='nr')
            b, extra_time, extra_moves = play_game(neural_net, config, mode='rn')
            raw_net_score_r += 1 + a - b
            a, extra_time, extra_moves = play_game(neural_net, config, mode='nm', state_dict=state_dict)
            b, extra_time, extra_moves = play_game(neural_net, config, mode='mn', state_dict=state_dict)
            raw_net_score_m += 1 + a - b
            net_time += time() - start_time
        except TimeoutError:
            print('timed out')

    neural_net_value_preds = [0]
    neural_net_policy_preds = [[0]]
    average_difference = 0
    percentage_acc = 0

    boards = []
    input_data = []
    mm_values = []
    mm_policies = []
    for i in range(100):
        game = Game()
        start_time = time()
        while not game.terminal():
            boards.append(game.board)
            input_data.append(game.make_image())
            mm_values.append((2 * value_dict[make_dict_key(game.board)] - 1) * (-1 if game.to_play == 0 else 1))
            mm_policies.append(state_dict[make_dict_key(game.board)])
            move = get_random_move(game)
            game.apply(move)
            if time() - start_time > 1:
                draw_board(game.board)
                break
    print('Random games done')
    tries = 0
    while tries < 3:
        try:
            neural_net_preds = neural_net.predict(array(input_data))
            neural_net_value_preds = list(neural_net_preds)[0]
            neural_net_policy_preds = list(neural_net_preds)[1]
            difference_sum = sum(abs(neural_net_value_preds[i] - mm_values[i]) for i in range(len(mm_values)))
            average_difference = difference_sum[0]/len(mm_values)
            num_correct = sum(1 if mm_policies[neural_net_policy_preds[i].index(max(neural_net_policy_preds[i]))] > 0 else
                              0 for i in range(len(neural_net_policy_preds)))
            percentage_acc = 100 * num_correct/len(mm_policies)
            tries = -1
        except TimeoutError:
            print('timed out... trying again')
            tries += 1
    print('Success rate vs random:', str(100 * score_r / config.checkpoint_game_num) + '%')
    print('Success rate vs perfect:', str(100 * score_m / config.checkpoint_game_num) + '%')
    print()
    print('Raw net success rate vs random:', str(100 * raw_net_score_r / config.checkpoint_game_num) + '%')
    print('Raw net success rate vs perfect:', str(100 * raw_net_score_m / config.checkpoint_game_num) + '%')
    print()
    if tries == -1:
        print('Policy percentage accuracy:', percentage_acc)
        print('Average value difference:', average_difference)
        print('Average value prediction:', average(neural_net_value_preds))
        print('Standard deviation of value prediction:', std(neural_net_value_preds))
        print('Example value predictions:', round(neural_net_value_preds[:10], 3))
        print('Example policy predictions:', round(neural_net_policy_preds[:5], 3))
        print('Time per move:', move_time/moves)
    print('Time taken for games with search:', search_time)
    print('Time taken for games without search:', net_time)
    print('Time:', ctime(time()))
    f = open('Stats', 'r')
    lines = f.readlines()
    f.close()
    f = open('Stats', 'w')
    for line in lines:
        f.write(line)
    f.write('Success rate vs random: ' + str(100 * score_r / config.checkpoint_game_num) + '%\n')
    f.write('Success rate vs perfect: ' + str(100 * score_m / config.checkpoint_game_num) + '%\n')
    f.write('\n')
    f.write('Raw net success rate vs random: ' + str(100 * raw_net_score_r / config.checkpoint_game_num) + '%\n')
    f.write('Raw net success rate vs perfect: ' + str(100 * raw_net_score_m / config.checkpoint_game_num) + '%\n')
    f.write('\n')
    f.write('Policy percentage accuracy: ' + str(percentage_acc) + '%\n')
    f.write('Average value difference: ' + str(average_difference))
    f.write('\n')
    f.close()


def test_minimax():
    s = [0, 0]
    w = [0, 0]
    d = [0, 0]
    l = [0, 0]
    mov = 0
    tim = 0
    game = Game()
    state_dict = {}
    value_dict = {}
    minimax(game, state_dict, value_dict)
    for i in range(1000000):
        res, mov_tim, movn = play_game('', '', mode='mr', state_dict=state_dict)
        mov += movn
        tim += mov_tim
        s[0] += res
        if res == 1:
            w[0] += 1
        elif res == 0.5:
            d[0] += 1
        else:
            l[0] += 1
        res, mov_tim, movn = play_game('', '', mode='rm', state_dict=state_dict)
        mov += movn
        tim += mov_tim
        s[1] += 1 - res
        if res == 0:
            w[1] += 1
        elif res == 0.5:
            d[1] += 1
        else:
            l[1] += 1

        if (i + 1) % 100000 == 0:
            print('Number of games:', i + 1)
            print('Total score:', s[0] + s[1], '(' + str(100 * (s[0] + s[1])/(2 * i + 1)) + '%)')
            print('Scores:', str(100 * s[0]/(i + 1)) + '% and', str(100 * s[1]/(i + 1)) + '%')
            print('Wins:', str(100 * w[0]/(i + 1)) + '% and', str(100 * w[1]/(i + 1)) + '%')
            print('Draws:', str(100 * d[0]/(i + 1)) + '% and', str(100 * d[1]/(i + 1)) + '%')
            print('Losses:', str(100 * l[0]/(i + 1)) + '% and', str(100 * l[1]/(i + 1)) + '%')
            try:
                print('Average move time:', tim/mov)
            except ZeroDivisionError:
                pass
            print('Time:', ctime(time()))
            print()
            print()
