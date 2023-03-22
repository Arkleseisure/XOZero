import sys
print(sys.version)
from Game import Game
from Make_image import make_output_image
from Net_move_maker import get_move
from Play_game import play_game
from time import time
from numpy import array, std, average, round
from random import random, randint
from copy import deepcopy
from Bits_and_Pieces import make_dict_key
from Move_maker import get_random_move
if __name__ == '__main__':
    from Neural_Network import make_net
    from Flipping import generate_flipped_data
    from Bits_and_Pieces import to_time, draw_board
    from Configuration import Config
    from Move_maker import minimax
    from matplotlib import pyplot as plt
    from multiprocessing import Pool, context
    from time import ctime
    from tensorflow.compat.v1 import Session, ConfigProto, GPUOptions
    from tensorflow.compat.v1.keras.backend import set_session
    from tensorflow import nn
    from keras.models import load_model
    import wandb

    hyperparameter_defaults = {
        'batch_size': 1024,
        'batch_number': 1,
        'epochs': 1,
        'neural_net_blocks': 1,
        'num_simulations': 50,
        'root_dirichlet_alpha': 0.3,
        'root_exploration_fraction': 0.25,
        'pb_c_base': 19652,
        'pb_c_init': 1.25
    }
    wandb.init(project='uncategorized', config=hyperparameter_defaults)
    wandb_config = wandb.config

    tf_config_memory_control = ConfigProto(gpu_options=GPUOptions(allow_growth=True,
                                                                  per_process_gpu_memory_fraction=0.1))
    session = Session(config=tf_config_memory_control)
    set_session(session)


def initializing_func(config, previous_net=False):
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
    tf_config = ConfigProto(gpu_options=GPUOptions(per_process_gpu_memory_fraction=
                                                   m_config.max_mem / m_config.processes,
                                                   allow_growth=True))
    m_session = Session(config=tf_config)
    set_session(m_session)

    global m_neural_net
    m_neural_net = load_model(m_config.net_name, custom_objects={'softmax_cross_entropy_with_logits_v2':
                                                                 nn.softmax_cross_entropy_with_logits})

    global prev_nn
    prev_nn = None
    if previous_net:
        prev_nn = load_model('prev net', custom_objects={'softmax_cross_entropy_with_logits_v2':
                                                         nn.softmax_cross_entropy_with_logits})


def test_games(state_dict):
    search_time = 0
    net_time = 0
    prev_time = 0

    # matches are labelled as follows: s = search, n = raw network, p = vs prev network
    start_time = time()
    sa, extra_time, extra_moves = play_game(m_neural_net, m_config, mode='sr')
    sb, extra_time, extra_moves = play_game(m_neural_net, m_config, mode='rs')
    sc, extra_time, extra_moves = play_game(m_neural_net, m_config, mode='sm', state_dict=state_dict)
    sd, extra_time, extra_moves = play_game(m_neural_net, m_config, mode='ms', state_dict=state_dict)
    search_time += time() - start_time

    start_time = time()
    na, extra_time, extra_moves = play_game(m_neural_net, m_config, mode='nr')
    nb, extra_time, extra_moves = play_game(m_neural_net, m_config, mode='rn')
    nc, extra_time, extra_moves = play_game(m_neural_net, m_config, mode='nm', state_dict=state_dict)
    nd, extra_time, extra_moves = play_game(m_neural_net, m_config, mode='mn', state_dict=state_dict)
    net_time += time() - start_time

    pa = 0
    pb = 0
    pc = 0
    pd = 0
    if prev_nn:
        start_time = time()
        pa, extra_time, extra_moves = play_game(m_neural_net, m_config, mode='nN', neural_net_2=prev_nn, noise=True)
        pb, extra_time, extra_moves = play_game(m_neural_net, m_config, mode='Nn', neural_net_2=prev_nn, noise=True)
        pc, extra_time, extra_moves = play_game(m_neural_net, m_config, mode='sS', neural_net_2=prev_nn, noise=True)
        pd, extra_time, extra_moves = play_game(m_neural_net, m_config, mode='Ss', neural_net_2=prev_nn, noise=True)
        prev_time += time() - start_time

    return array([sa, sb, sc, sd]), array([na, nb, nc, nd]), array([pa, pb, pc, pd]), \
           array([search_time, net_time, prev_time])


def predict(data):
    return m_neural_net.predict(array(data))


def self_play_game(*args):
    input_data = []
    target_policy_data = []
    target_value_data = []
    game = Game()
    next_root = ''
    result = False
    while not result:
        input_data.append(game.make_image())
        move, search_stats, next_root = get_move(game, m_neural_net, m_config, next_root, self_play=True)
        target_policy_data.append(make_output_image(search_stats, len(game.board[0])))

        game.apply(move)

        if game.num_moves > 9:
            print(game.legal_moves, game.board)
        result = game.terminal()

    for i in range(game.num_moves):
        target_value_data.append(0 if result == 'full' else 2 * ((game.num_moves - i) % 2) - 1)

    return [input_data, target_value_data, target_policy_data]


if __name__ == '__main__':
    # makes data based off perfect play minimax data
    def make_minimax_data(args):
        state_dict = args[0]
        value_dict = args[1]
        input_data = []
        target_value_data = []
        target_policy_data = []
        game = Game()
        a = []
        for i in range(9):
            a.append(0)
        while not game.terminal():
            input_data.append(game.make_image())
            target_value_data.append((2 * value_dict[make_dict_key(game.board)] - 1) * (-1 if game.to_play == 0 else 1))
            target_policy_data.append(deepcopy(a))
            for item in state_dict[make_dict_key(game.board)]:
                target_policy_data[-1][int(item[0]) + 3 * int(item[1])] = 1 / len(state_dict[make_dict_key(game.board)])
            move = get_random_move(game)
            game.apply(move)
        return [input_data, target_value_data, target_policy_data]

    # calculates the elo rating from a score vs an opponent
    def calc_elo(score, games, opp_elo):
        return opp_elo + (400 * (score - games/2))/games


    def test_network(neural_net, config, prev_net, net_est_elo, search_est_elo):
        org_start_time = time()
        print('Started Testing')
        search_scores = array([0.0, 0.0, 0.0, 0.0])
        net_scores = array([0.0, 0.0, 0.0, 0.0])
        prev_scores = array([0.0, 0.0, 0.0, 0.0])
        times = array([0.0, 0.0, 0.0])
        num_games = config.checkpoint_game_num // 2
        state_dict = {}
        value_dict = {}
        minimax(state_dict, value_dict)

        # plays games vs random, minimax and last network to be tested
        with Pool(config.processes, initializer=initializing_func, initargs=[config, prev_net]) as pool:
            data = pool.imap_unordered(test_games, [state_dict for i in range(num_games)])
            for i in range(num_games):
                if (i + 1) % 5 == 0:
                    print('Set:', (i + 1) * 2)
                try:
                    item = data.next(timeout=20 if i == 0 else 10)
                    search_scores += item[0]
                    net_scores += item[1]
                    prev_scores += item[2]
                    times += item[3]
                    i += 1
                except context.TimeoutError:
                    print('Testing time out')
                    print('Not all games have been completed')

            # tests the accuracy of the raw network
            input_data = []
            mm_values = []
            mm_policies = []
            rand_values = []
            rand_policies = []
            for i in range(num_games):
                game = Game()
                start_time = time()
                while not game.terminal():
                    input_data.append(game.make_image())
                    mm_values.append((2 * value_dict[make_dict_key(game.board)] - 1) * (-1 if game.to_play == 1 else 1))
                    mm_policies.append(state_dict[make_dict_key(game.board)])
                    rand_values.append(random() * 2 - 1)
                    rand_policies.append(str(randint(0, 2)) + str(randint(0, 2)))
                    move = get_random_move(game)
                    game.apply(move)
                    if time() - start_time > 1:
                        draw_board(game.board)
                        break

            neural_net_value_preds = [0]
            neural_net_policy_preds = [[0]]
            average_difference = 0
            percentage_acc = 0
            rand_diff = 0
            rand_acc = 0
            print('Random games done')
            tries = 0
            while tries < 3:
                try:
                    neural_net_preds = pool.imap_unordered(predict, [input_data])
                    neural_net_preds = neural_net_preds.next(timeout=5)
                    neural_net_value_preds = list(neural_net_preds[0])
                    neural_net_policy_preds = []
                    for item in neural_net_preds[1]:
                        neural_net_policy_preds.append(list(item))
                    difference_sum = sum(
                        abs(neural_net_value_preds[i][0] - mm_values[i]) for i in range(len(mm_values)))
                    average_difference = difference_sum / len(mm_values)
                    rand_diff = sum(abs(rand_values[i] - mm_values[i]) for i in range(len(mm_values))) / len(
                        mm_values)

                    # mm policies holds the possible correct moves in the situation in the form 'xy'.
                    # This converts the index of the maximum predicted value into that format and adds one if
                    # it is a valid move, and does this for each predicted value.
                    num_correct = sum(1 if str(neural_net_policy_preds[i].index(max(neural_net_policy_preds[i])) % 3) +
                                      str(neural_net_policy_preds[i].index(max(neural_net_policy_preds[i])) // 3)
                                      in mm_policies[i] else 0 for i in range(len(mm_policies)))
                    percentage_acc = 100 * num_correct / len(mm_policies)
                    rand_acc = 100 * sum(1 if rand_policies[i] in mm_policies[i] else 0 for
                                         i in range(len(mm_policies))) / len(mm_policies)
                    tries = 4
                except context.TimeoutError:
                    print('timed out... trying again')
                    tries += 1

        # calculating results
        score_r = search_scores[0] + num_games - search_scores[1]
        score_m = search_scores[2] + num_games - search_scores[3]
        raw_net_score_r = net_scores[0] + num_games - net_scores[1]
        raw_net_score_m = net_scores[2] + num_games - net_scores[3]
        score_pn = prev_scores[0] + num_games - prev_scores[1]
        score_ps = prev_scores[2] + num_games - prev_scores[3]
        search_time = times[0]
        net_time = times[1]
        prev_time = times[2]

        net_elo = calc_elo(raw_net_score_m + raw_net_score_r, 2 * config.checkpoint_game_num, 175)
        search_elo = calc_elo(score_m + score_r, 2 * config.checkpoint_game_num, 175)
        if prev_net:
            net_est_elo = calc_elo(score_pn, config.checkpoint_game_num, net_est_elo)
            search_est_elo = calc_elo(score_ps, config.checkpoint_game_num, search_est_elo)
        else:
            net_est_elo = calc_elo(raw_net_score_r, config.checkpoint_game_num, net_est_elo)
            search_est_elo = calc_elo(score_r, config.checkpoint_game_num, search_est_elo)

        print('Success rate vs random:', str(100 * score_r / config.checkpoint_game_num) + '%')
        print('Success rate vs perfect:', str(100 * score_m / config.checkpoint_game_num) + '%')
        print('Success rate vs previous:', str(100 * score_ps / config.checkpoint_game_num) + '%')
        print()
        print('Raw net success rate vs random:', str(100 * raw_net_score_r / config.checkpoint_game_num) + '%')
        print('Raw net success rate vs perfect:', str(100 * raw_net_score_m / config.checkpoint_game_num) + '%')
        print('Raw net success rate vs previous:', str(100 * score_pn / config.checkpoint_game_num) + '%')
        print()
        if tries == 4:
            print('Policy percentage accuracy:', percentage_acc)
            print('Random percentage accuracy:', rand_acc)
            print('Average value difference:', average_difference)
            print('Random value difference:', rand_diff)
            print()
            print('Average value prediction:', average(neural_net_value_preds))
            print('Standard deviation of value prediction:', std(neural_net_value_preds))
            print('Example value predictions:', round(array(neural_net_value_preds[:10]), 3))
            print('Example policy predictions:', round(array(neural_net_policy_preds[:5]), 3))
            print()
        print('Time taken for games with search:', search_time)
        print('Time taken for games without search:', net_time)
        print('Time taken for games with previous network:', prev_time)
        print('Total time taken for testing:', time() - org_start_time)
        print('Time:', ctime(time()))
        f = open('Stats', 'r')
        lines = f.readlines()
        f.close()
        f = open('Stats', 'w')
        for line in lines:
            f.write(line)
        f.write('Success rate vs random: ' + str(100 * score_r / config.checkpoint_game_num) + '%\n')
        f.write('Success rate vs perfect: ' + str(100 * score_m / config.checkpoint_game_num) + '%\n')
        f.write('Success rate vs previous net: ' + str(100 * score_ps / config.checkpoint_game_num) + '%\n')
        f.write('\n')
        f.write('Raw net success rate vs random: ' + str(100 * raw_net_score_r / config.checkpoint_game_num) + '%\n')
        f.write('Raw net success rate vs perfect: ' + str(100 * raw_net_score_m / config.checkpoint_game_num) + '%\n')
        f.write('Raw net success rate vs previous net: ' + str(100 * score_pn / config.checkpoint_game_num) + '%\n')
        f.write('\n')
        f.write('Policy percentage accuracy: ' + str(percentage_acc) + '%\n')
        f.write('Average value difference: ' + str(average_difference) + '\n')
        f.write('\n\n')
        f.close()
        neural_net.save('prev net')
        wandb.log({'Search v random': 100 * score_r / config.checkpoint_game_num,
                   'Search v perfect': 100 * score_m / config.checkpoint_game_num,
                   'Search v previous': 100 * score_ps / config.checkpoint_game_num,
                   'Net v random': 100 * raw_net_score_r / config.checkpoint_game_num,
                   'Net v perfect': 100 * raw_net_score_m / config.checkpoint_game_num,
                   'Net v previous': 100 * score_pn / config.checkpoint_game_num,
                   'Search elo': search_elo,
                   'Net elo': net_elo,
                   'Search estimated elo': search_est_elo,
                   'Net estimated elo': net_est_elo,
                   'Policy accuracy': percentage_acc,
                   'Value difference': average_difference}, commit=False)
        return net_est_elo, search_est_elo


def main(steps_done, start_time):
    if __name__ == '__main__':
        state_dict = {}
        value_dict = {}
        minimax(state_dict, value_dict)
        play_game('', state_dict=state_dict, mode='mh', config='')
        print('main started')
        config = Config(wandb_config)
        if steps_done == 0:
            neural_net = make_net(config.neural_net_blocks)
            neural_net.save(config.net_name)
        else:
            neural_net = load_model(config.net_name, custom_objects={'softmax_cross_entropy_with_logits_v2':
                                                                     nn.softmax_cross_entropy_with_logits})
        positions = config.step_size * config.batch_number // 8
        num_funcs = positions // 5 + 1
        state_dict = {}
        value_dict = {}
        minimax(state_dict, value_dict)
        extra_id = []
        extra_vd = []
        extra_pd = []
        flipping_time = 0
        data_gen_time = 0
        training_time = 0
        testing_time = 0
        net_est_elo = 0
        search_est_elo = 0
        print('Time started')
        with Pool(config.processes, initializer=initializing_func, initargs=[config]) as pool:
            for i in range(config.training_steps//config.batch_number - steps_done):
                if (i + steps_done) * config.batch_number % config.checkpoint_interval == 0:
                    start_time_2 = time()
                    net_est_elo, search_est_elo = test_network(neural_net, config,
                                                               prev_net=False,  # if i == 0 else True
                                                               net_est_elo=net_est_elo, search_est_elo=search_est_elo)
                    testing_time += time() - start_time_2

                print()
                print('Training step:', (i + steps_done) * config.batch_number + 1)
                # '''
                input_data = extra_id
                target_value_data = extra_vd
                target_policy_data = extra_pd
                start_time_2 = time()
                while len(input_data) < positions:
                    data = pool.imap_unordered(self_play_game, range(num_funcs), chunksize=3)
                    try:
                        for m in data:
                            input_data.extend(m[0])
                            target_value_data.extend(m[1])
                            target_policy_data.extend(m[2])
                    except context.TimeoutError:
                        print('timed out')
                    except StopIteration:
                        pass

                # pool.close()
                '''
                input_data = []
                target_value_data = []
                target_policy_data = []
                processes_completed = 0
                mems_completed = 0
                timing_nbr = 50
                positions = config.step_size * config.batch_number // 8
                num_funcs = positions // 5 + 1
                print(num_funcs, 'functions')
                for j in range(10 - processes_completed):
                    config.processes = j + processes_completed + 1
                    for k in range((10 - mems_completed) if j == 0 else 10):
                        start_time = time()
                        config.max_mem = (k + 1 + (mems_completed if j == 0 else 0)) / 10
                        with Pool(config.processes, initializer=initializing_func) as pool:
                            for l in range(timing_nbr):
                                input_data = input_data[positions:]
                                target_value_data = target_value_data[positions:]
                                target_policy_data = target_policy_data[positions:]
                                while len(input_data) < positions:
                                    data = pool.imap_unordered(play_multiprocessed_games, [times for m in range(num_funcs)])
                                    for m in range(num_funcs):
                                        try:
                                            next_data = data.next(timeout=25 if m == 0 else 5)
                                            input_data.extend(next_data[0])
                                            target_value_data.extend(next_data[1])
                                            target_policy_data.extend(next_data[2])
                                        except context.TimeoutError:
                                            print('timed out')
                                        except StopIteration:
                                            break
                                print(l + 1, 'complete')
                                print(len(input_data), 'positions generated')
                            pool.close()
                        print('Avg time taken to generate data for', config.processes, 'processes with max', config.max_mem,
                              'memory for processes :', (time() - start_time)/timing_nbr)
                        print('Time:', ctime(time()))
                        f = open('Stats', 'r')
                        lines = f.readlines()
                        f.close()
                        f = open('Stats', 'w')
                        for line in lines:
                            f.write(line)
                        f.write(str((time() - start_time) / timing_nbr) + '\n')
                        f.close()
                # '''
                extra_id = input_data[positions:]
                extra_pd = target_policy_data[positions:]
                extra_vd = target_value_data[positions:]
                input_data = input_data[:positions]
                target_value_data = target_value_data[:positions]
                target_policy_data = target_policy_data[:positions]
                data_gen_time += time() - start_time_2
                print(target_value_data[:10])
                print(target_policy_data[:10])
                start_time_2 = time()
                generate_flipped_data(input_data, target_value_data, target_policy_data)
                flipping_time += time() - start_time_2
                start_time_2 = time()
                hist = neural_net.fit(array(input_data), [array(target_value_data), array(target_policy_data)],
                                      batch_size=config.batch_size, epochs=config.epochs,
                                      validation_split=config.val_split, verbose=2)
                wandb.log({'value_acc': hist.history['value_output_acc'][0],
                           'policy_acc': hist.history['policy_output_acc'][0],
                           'loss': hist.history['loss'][0]})
                training_time += time() - start_time_2
                time_taken = time() - start_time
                time_left = time_taken * (config.training_steps - (steps_done + i + 1) * config.batch_number) \
                        / ((steps_done + i + 1) * config.batch_number)
                print('Average flipping time:', flipping_time/(steps_done + i + 1))
                print('Average data generation time:', data_gen_time/(steps_done + i + 1))
                print('Average training time:', training_time/(steps_done + i + 1))
                print('Average testing time:', testing_time/(steps_done + i + 1))
                print('Time taken:', to_time(time_taken))
                print('Time left:', to_time(time_left))
                print('Estimated finishing time:', ctime(time() + time_left))
                print('Time:', ctime(time()))
                neural_net.save(config.net_name)
            start_time_2 = time()
            test_network(neural_net, config, prev_net=False,  # True,
                         net_est_elo=net_est_elo, search_est_elo=search_est_elo)
            testing_time += time() - start_time_2
            time_taken = time() - start_time
            print('Average flipping time:', flipping_time / (config.training_steps//config.batch_number - steps_done))
            print('Average data generation time:', data_gen_time / (config.training_steps//config.batch_number -
                                                                    steps_done))
            print('Average training time:', training_time / (config.training_steps//config.batch_number - steps_done))
            print('Average testing time:', testing_time / (config.training_steps//config.batch_number - steps_done))
            print('Time taken:', to_time(time_taken))
            print('Time:', ctime(time()))
            plt.pie([data_gen_time, testing_time, training_time, flipping_time],
                    labels=['data gen', 'testing', 'training', 'flipping'])
            wandb.log({'pie_chart': wandb.Image(plt)})
            return -1


if __name__ == '__main__':
    steps_done = 0
    start_time = time()
    f = open('Stats', 'w')
    f.close()
    while not steps_done == -1:
        steps_done = main(steps_done, start_time)
        print('exited')
        print('Steps done:', steps_done)

# random elo = 0, perfect elo = 467, middle = 233
