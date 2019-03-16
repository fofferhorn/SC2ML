import os
from multiprocessing import Process, Value

from absl import app
from absl import flags
import sys
import time

import numpy as np
from scipy import sparse

import pysc2
from pysc2 import run_configs
from s2clientprotocol import sc2api_pb2 as sc_pb

from websocket import WebSocketTimeoutException

import constants as c

FLAGS = flags.FLAGS

# General settings
flags.DEFINE_string(name = 'replays_path', default = 'filtered_replays', help = 'The path to the replays to extract actions from the current directory.')
flags.DEFINE_string(name = 'save_path', default = 'extracted_actions', help = 'The path to the folder to save the replays in from the current directory.')
flags.DEFINE_integer(name = 'n_instance', default = 4, help = 'The default amount of threads to use to filter the replays.')
flags.DEFINE_integer(name = 'batch_size', default = 10, help = 'The amount of replays each worker process takes at a time.')
flags.DEFINE_integer(name = 'step_mul', default = 1, help = 'The amount of game steps between each observation.')
flags.DEFINE_integer(name = 'start_from_replay', default = 670, help = 'The replay number to start from.')


FLAGS(sys.argv)


def extract_actions(counter, replays_path, save_path, batch_size, step_mul):
    # Check if the replays_path exists
    if not os.path.isdir(replays_path):
        raise ValueError('The path ' + replays_path + ' does not exist.')

    # Make list of the paths to all the replays
    cwd = os.getcwd()
    replay_paths = []
    for replay in os.listdir(replays_path):
        replay_path = os.path.join(cwd, replays_path, replay)
        if os.path.isfile(replay_path) and replay.lower().endswith('.sc2replay'):
            replay_paths.append(replay_path)

    # Check if the save_path exists. Otherwise we need to create it
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # Used for picking up from where an exception was thrown.
    replays_range_left_in_batch = (0,0)

    while True:
        try:
            run_config = run_configs.get()
            with run_config.start() as controller:
                while True:
                    # Variables for later
                    batch_start = 0
                    batch_end = 0

                    # Check if we resumed here from an exception
                    if replays_range_left_in_batch == (0,0):
                        # Everything is good
                        with counter.get_lock():
                            if counter.value * batch_size > len(replay_paths):
                                print('                                                    Reached the end of the replay list. Returning...')
                                return
                            batch_number = counter.value
                            counter.value += 1

                        batch_start = batch_number * batch_size
                        batch_end = batch_number * batch_size + batch_size
                        if batch_end > len(replay_paths) - 1:
                            batch_end = len(replay_paths) - 1
                    else:
                        # We resumed here from an exception. Skip the replay that caused an exception.
                        batch_start = replays_range_left_in_batch[0] + 1
                        batch_end = replays_range_left_in_batch[1]
                        print('                                                    Resuming with batch from a crash. Resuming from ' + str(batch_start) + ' to ' + str(batch_end))

                    for index in range(batch_start, batch_end):
                        print('================================================================================ Processing replay #' + str(index + 1))

                        replays_range_left_in_batch = (index, batch_end)

                        replay_path = replay_paths[index]

                        replay_data = run_config.replay_data(replay_path)
                        info = controller.replay_info(replay_data)


                        interface = sc_pb.InterfaceOptions()
                        interface.raw = True
                        interface.score = False

                        for player in info.player_info:
                            player_game_data_points = []
                            
                            start_replay = sc_pb.RequestStartReplay(
                                replay_data = replay_data,
                                options = interface,
                                disable_fog = False,
                                observed_player_id = player.player_info.player_id)

                            controller.start_replay(start_replay)

                            controller.step()

                            time_step = 0

                            try:
                                while True:
                                    time_step += step_mul
                                    controller.step(step_mul)
                                    obs = controller.observe()

                                    for action in obs.actions:
                                        mapped_action = get_macro_action(action)
                                        if mapped_action is not None:
                                            resources = get_resources(obs.observation)
                                            upgrades = get_upgrades(obs.observation.raw_data.player.upgrade_ids)
                                            in_progress = get_units_in_progress(obs.observation.raw_data.units)
                                            friendly_unit_list = get_friendly_unit_list(obs.observation.raw_data.units)
                                            enemy_unit_list = get_enemy_unit_list(obs.observation.raw_data.units)

                                            new_data_point = []
                                            new_data_point.append(time_step)        # 1
                                            new_data_point += resources             # 9
                                            new_data_point += upgrades              # 26
                                            new_data_point += in_progress           # 70
                                            new_data_point += friendly_unit_list    # 44
                                            new_data_point += enemy_unit_list       # 44
                                            new_data_point += mapped_action    # 54

                                            player_game_data_points.append(new_data_point)

                                    # The game has finished if there is a player_result
                                    if obs.player_result:
                                        print('                                                    Replay #' + str(index+1) + ' from player ' + str(player.player_info.player_id) + '\'s perspective has finished.')
                                        break
                            except KeyboardInterrupt:
                                return
                            
                            # Save the data for this player.
                            replay_save_path = os.path.join(cwd, save_path, replay_path.split('/')[-1].split('.')[0] + '_' + str(player.player_info.player_id))
                            save_replay_data(replay_save_path, player_game_data_points)
                        
                        print('                                                    Finished processing game #' + str(index + 1) + '.')

                    # Everything went smothely with this replay batch. Reset the counter and move on.
                    replays_range_left_in_batch = (0,0)

        except KeyboardInterrupt:
            return
        except WebSocketTimeoutException:
            print('                                                    Websocket timed out.')
        except pysc2.lib.protocol.ConnectionError:
            print('                                                    Websocket timed out.')
        except:
            print('                                                    Something went wrong. Skipping this replay.')


def get_resources(observation):
    resources = [
        observation.player_common.minerals,
        observation.player_common.vespene,
        observation.player_common.food_cap,
        observation.player_common.food_used,
        observation.player_common.food_army,
        observation.player_common.food_workers,
        observation.player_common.idle_worker_count,
        observation.player_common.army_count,
        observation.player_common.warp_gate_count
    ]

    return resources


def get_units_in_progress(units):
    # Number of each building, unit and upgrade in progress
    in_progress_list = [0] * (44 + 26)

    for unit in units:
        if unit.alliance == 1:
            # Something is being built/something in the map. E.g. a building being built.
            if unit.build_progress < 1:
                protoss_unit = c.protoss_unit_mapper.get(unit.unit_type)
                in_progress_list[protoss_unit] += 1

            # Something is being built/something by something else. E.g. a building training a unit.
            if unit.orders is not None:
                for order in unit.orders:
                    # The unit being built by e.g. a building.
                    in_progress_entity = c.protoss_action_to_unit_mapper.get(order.ability_id) 
        
                    if in_progress_entity is not None:
                        in_progress_list[in_progress_entity] += 1

    return in_progress_list


def get_friendly_unit_list(units):
    # Amount of units for each protoss unit
    unit_list = [0] * 44

    for unit in units:
        if unit.alliance == 1:
            protoss_unit = c.protoss_unit_mapper.get(unit.unit_type)
            if protoss_unit is not None:
                unit_list[protoss_unit] += 1

    return unit_list


def get_enemy_unit_list(units):
    # Amount of units for each protoss unit
    unit_list = [0] * 44

    for unit in units:
        if unit.alliance == 4:
            unit_index = c.protoss_unit_mapper.get(unit.unit_type)
            if unit_index is not None:
                unit_list[unit_index] += 1

    return unit_list


def get_upgrades(upgrade_ids):
    upgrades = [0] * 26

    for upgrade_id in upgrade_ids:
        protoss_upgrade = c.protoss_upgrade_mapper.get(upgrade_id)
        if protoss_upgrade is not None:
            upgrades[protoss_upgrade] += 1
    
    return upgrades


def get_macro_action(action):
    actions = [0] * 54
    if hasattr(action, "action_raw") and hasattr(action.action_raw, "unit_command"):
        mapped_action = c.protoss_action_to_unit_mapper.get(action.action_raw.unit_command.ability_id)

        if mapped_action is not None:
            actions[mapped_action] += 1
            return actions
    
    return None


def save_replay_data(save_path, replay_data):
    np.save(save_path, replay_data)


def main(argv):
    # The list of processes
    jobs = []              

    # Lock so the processes don't work on the same replays
    counter = Value('i', int(FLAGS.start_from_replay/FLAGS.batch_size))     

    for _ in range(FLAGS.n_instance):
        p = Process(target = extract_actions, args = [
            counter, 
            FLAGS.replays_path, 
            FLAGS.save_path, 
            FLAGS.batch_size,
            FLAGS.step_mul])
        jobs.append(p)
        p.daemon = True
        p.start()

        time.sleep(1)

    # Wait for each process to finish.
    for i in range(FLAGS.n_instance):
        jobs[i].join()


if __name__ == "__main__":
    # counter = Value('i', int(FLAGS.start_from_replay/FLAGS.batch_size))     # Lock so the processes don't work on the same replays
    # extract_actions(counter, FLAGS.replays_path, FLAGS.save_path, FLAGS.batch_size, FLAGS.step_mul)

    app.run(main)