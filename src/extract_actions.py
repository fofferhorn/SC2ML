import os
from multiprocessing import Process, Value

from absl import app
from absl import flags
import sys

import numpy as np
from scipy import sparse

from pysc2 import run_configs
from s2clientprotocol import sc2api_pb2 as sc_pb

FLAGS = flags.FLAGS

# General settings
flags.DEFINE_string(name = 'replays_path', default = 'filtered_replays', help = 'The path to the replays to extract actions from the current directory.')
flags.DEFINE_string(name = 'save_path', default = 'extracted_actions', help = 'The path to the folder to save the replays in from the current directory.')
flags.DEFINE_integer(name = 'n_instance', default = 8, help = 'The default amount of threads to use to filter the replays.')
flags.DEFINE_integer(name = 'batch_size', default = 10, help = 'The amount of replays each worker process takes at a time.')
flags.DEFINE_integer(name = 'step_mul', default = 1, help = 'The amount of game steps between each observation.')


FLAGS(sys.argv)


# Mapping of unit_id to index in list.
protoss_unit_mapper = {
        # Units
        84: 0,      # Probe
        73: 1,      # Zealot
        77: 2,      # Sentry
        311: 3,     # Adept
        74: 4,      # Stalker
        75: 5,      # HighTemplar
        76: 6,      # DarkTemplar
        141: 7,     # Archon
        4: 8,       # Colossus
        694: 9,     # Disruptor
        82: 10,     # Observer
        83: 11,     # Immortal
        81: 12,     # WarpPrism
        78: 13,     # Phoenix
        495: 14,    # Oracle
        80: 15,     # VoidRay
        496: 16,    # Tempest
        79: 17,     # Carrier
        85: 18,     # Interceptor
        10: 19,     # Mothership
        488: 20,    # MothershipCore

        # Buildings
        59: 21,     # Nexus
        60: 22,     # Pylon
        61: 23,     # Assimilator
        62: 24,     # Gateway
        133: 25,    # WarpGate
        72: 26,     # CyberneticsCore
        65: 27,     # TwilightCouncil
        68: 28,     # TemplarArchive
        69: 29,     # DarkShrine
        63: 30,     # Forge
        66: 31,     # PhotonCannon
        1910: 32,   # ShieldBattery
        70: 33,     # RoboticsBay
        71: 34,     # RoboticsFacility
        67: 35,     # Stargate
        64: 36,     # FleetBeacon

        # Abilities
        801: 37,    # AdeptPhaseShift
        135: 38,    # ForceField
        1911: 39,   # ObserverSurveillanceMode
        733: 40,    # DisruptorPhased
        136: 41,    # WarpPrismPhasing
        894: 42,    # PylonOvercharged
        732: 43     # StasisTrap
}

# Mapping of upgrades to indexes in list
protoss_upgrade_mapper = {
    1: 0,       # CARRIERLAUNCHSPEEDUPGRADE
    39: 1,      # PROTOSSGROUNDWEAPONSLEVEL1
    40: 2,      # PROTOSSGROUNDWEAPONSLEVEL2
    41: 3,      # PROTOSSGROUNDWEAPONSLEVEL3
    42: 4,      # PROTOSSGROUNDARMORSLEVEL1
    43: 5,      # PROTOSSGROUNDARMORSLEVEL2
    44: 6,      # PROTOSSGROUNDARMORSLEVEL3
    45: 7,      # PROTOSSSHIELDSLEVEL1
    46: 8,      # PROTOSSSHIELDSLEVEL2
    47: 9,      # PROTOSSSHIELDSLEVEL3
    48: 10,     # OBSERVERGRAVITICBOOSTER
    49: 11,     # GRAVITICDRIVE
    50: 12,     # EXTENDEDTHERMALLANCE
    52: 13,     # PSISTORMTECH
    78: 14,     # PROTOSSAIRWEAPONSLEVEL1
    79: 15,     # PROTOSSAIRWEAPONSLEVEL2
    80: 16,     # PROTOSSAIRWEAPONSLEVEL3
    81: 17,     # PROTOSSAIRARMORSLEVEL1
    82: 18,     # PROTOSSAIRARMORSLEVEL2
    83: 19,     # PROTOSSAIRARMORSLEVEL3
    84: 20,     # WARPGATERESEARCH
    86: 21,     # CHARGE
    87: 22,     # BLINKTECH
    99: 23,     # PHOENIXRANGEUPGRADE
    130: 24,    # ADEPTPIERCINGATTACK
    141: 25     # DARKTEMPLARBLINKUPGRADE
}

macro_actions = [
    "Build", 
    "Cancel", 
    "Morph", 
    "Research", 
    "Stop", 
    "Train", 
    "TrainWarp"
]


def extract_actions(counter, replays_path, save_path, batch_size, run_config):
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
    try:
        while True:
            with run_config.start() as controller:
                while True:
                    with counter.get_lock():
                        if counter.value * batch_size > len(replays_path):
                            return
                        i = counter.value
                        counter.value += 1

                    batch_start = i * batch_size
                    batch_end = i * batch_size + batch_size

                    if batch_end > len(replay_paths) - 1:
                        batch_end = len(replay_paths) - 1

                    for index in range(batch_start, batch_end):
                        print('================================================================================ Processing replay #' + str(index + 1))

                        replay_path = replay_paths[index]

                        replay_save_path = os.path.join(cwd, save_path, replay_path.split('/')[-1].split('.')[0])

                        replay_data = run_config.replay_data(replay_path)
                        info = controller.replay_info(replay_data)

                        data_points = []

                        interface = sc_pb.InterfaceOptions()
                        interface.raw = True
                        interface.score = False

                        for player in info.player_info:
                            start_replay = sc_pb.RequestStartReplay(
                                replay_data = replay_data,
                                options = interface,
                                disable_fog = False,
                                observed_player_id = player.player_info.player_id)

                            controller.start_replay(start_replay)

                            controller.step()

                            # All actions a player can take
                            abilities = controller.data_raw().abilities

                            steps = 1

                            try:
                                while True:
                                    steps += FLAGS.step_mul
                                    controller.step(FLAGS.step_mul)
                                    obs = controller.observe()

                                    new_data_points = []

                                    for action in obs.actions:
                                        if is_macro_action(action, abilities):

                                            resources = get_resources(obs.observation)
                                            upgrades = get_upgrades(obs.observation.raw_data.player.upgrade_ids)
                                            friendly_unit_list = get_friendly_unit_list(obs.observation.raw_data.units)
                                            enemy_unit_list = get_enemy_unit_list(obs.observation.raw_data.units)

                                            new_data_point = []
                                            new_data_point.append(resources)
                                            new_data_point.append(upgrades)
                                            new_data_point.append(friendly_unit_list)
                                            new_data_point.append(enemy_unit_list)
                                            new_data_point.append(player.player_result.result)
                                            new_data_point.append(action)

                                            new_data_points.append(new_data_point)

                                    data_points = data_points + new_data_points

                                    # The game has finished if there is a player_result
                                    if obs.player_result:
                                        save_replay_data(data_points, replay_save_path)

                            except KeyboardInterrupt:
                                return
    except KeyboardInterrupt:
        return
    except:
        print("Bad replay. Skipping batch.")


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


def get_friendly_unit_list(units):
    # Amount of units for each protoss unit
    unit_list = [0] * 44

    for unit in units:
        if unit.alliance == 1:
            protoss_unit = protoss_unit_mapper.get(unit.unit_type)
            if protoss_unit is not None:
                unit_list[protoss_unit] += 1

    return unit_list


def get_enemy_unit_list(units):
    # Amount of units for each protoss unit
    unit_list = [0] * 44

    for unit in units:
        if unit.alliance == 1:
            unit_index = protoss_unit_mapper.get(unit.unit_type)
            if unit_index is not None:
                unit_list[unit_index] += 1

    return unit_list


def get_upgrades(upgrade_ids):
    upgrades = [0] * 26

    for upgrade_id in upgrade_ids:
        protoss_upgrade = protoss_upgrade_mapper.get(upgrade_id)
        if protoss_upgrade is not None:
            upgrades[protoss_upgrade] += 1
    
    return upgrades


def is_macro_action(action, abilities):
    if hasattr(action, "action_raw") and hasattr(action.action_raw, "unit_command"):
        ability_type = abilities[action.action_raw.unit_command.ability_id].friendly_name.split(' ')[0]
        if ability_type in macro_actions:
            return True
    
    return False


def save_replay_data(save_path, replay_data):
    np.save(save_path, replay_data)


def main(argv):
    jobs = []                   # The list of processes
    counter = Value('i', 0)     # Lock so the processes don't work on the same replays
    run_config = run_configs.get()

    for _ in range(FLAGS.n_instance):
        p = Process(target = extract_actions, args = [
            counter, 
            FLAGS.replays_path, 
            FLAGS.save_path, 
            FLAGS.batch_size,
            run_config])
        jobs.append(p)
        p.daemon = True
        p.start()

    # Wait for each process to finish.
    for i in range(FLAGS.n_instance):
        jobs[i].join()


if __name__ == "__main__":
    counter = Value('i', 0)     # Lock so the processes don't work on the same replays
    run_config = run_configs.get()

    extract_actions(counter, FLAGS.replays_path, FLAGS.save_path, FLAGS.batch_size, run_config)

    #app.run(main)