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

FLAGS = flags.FLAGS

# General settings
flags.DEFINE_string(name = 'replays_path', default = 'filtered_replays', help = 'The path to the replays to extract actions from the current directory.')
flags.DEFINE_string(name = 'save_path', default = 'extracted_actions', help = 'The path to the folder to save the replays in from the current directory.')
flags.DEFINE_integer(name = 'n_instance', default = 4, help = 'The default amount of threads to use to filter the replays.')
flags.DEFINE_integer(name = 'batch_size', default = 10, help = 'The amount of replays each worker process takes at a time.')
flags.DEFINE_integer(name = 'step_mul', default = 1, help = 'The amount of game steps between each observation.')
flags.DEFINE_integer(name = 'start_from_replay', default = 750, help = 'The replay number to start from.')


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

# Mapping of macro actions to the buildings/units/upgrades they will turn into. 
protoss_action_to_unit_mapper = {
    # Build
    882: 23,    # BUILD_ASSIMILATOR
    894: 26,    # BUILD_CYBERNETICSCORE
    891: 29,    # BUILD_DARKSHRINE
    885: 36,    # BUILD_FLEETBEACON
    884: 30,    # BUILD_FORGE
    883: 24,    # BUILD_GATEWAY
    1042: 18,   # BUILD_INTERCEPTORS
    880: 21,    # BUILD_NEXUS
    887: 31,    # BUILD_PHOTONCANNON
    881: 22,    # BUILD_PYLON
    892: 33,   # BUILD_ROBOTICSBAY
    893: 34,   # BUILD_ROBOTICSFACILITY
    895: 32,   # BUILD_SHIELDBATTERY
    889: 35,   # BUILD_STARGATE
    890: 28,   # BUILD_TEMPLARARCHIVE
    886: 27,   # BUILD_TWILIGHTCOUNCIL

    # Morph
    1766: 7,  # MORPH_ARCHON
    1520: 24,  # MORPH_GATEWAY
    1847: 19,  # MORPH_MOTHERSHIP
    1518: 25,  # MORPH_WARPGATE

    # Research
    1594: 24,  # RESEARCH_ADEPTRESONATINGGLAIVES
    1593: 22,  # RESEARCH_BLINK
    1592: 21,  # RESEARCH_CHARGE
    1097: 12,  # RESEARCH_EXTENDEDTHERMALLANCE
    1093: 10,  # RESEARCH_GRAVITICBOOSTER
    1094: 11,  # RESEARCH_GRAVITICDRIVE
    44: 0,    # RESEARCH_INTERCEPTORGRAVITONCATAPULT
    46: 23,    # RESEARCH_PHOENIXANIONPULSECRYSTALS
    1565: 17,  # RESEARCH_PROTOSSAIRARMORLEVEL1
    1566: 18,  # RESEARCH_PROTOSSAIRARMORLEVEL2
    1567: 19,  # RESEARCH_PROTOSSAIRARMORLEVEL3
    1562: 14,  # RESEARCH_PROTOSSAIRWEAPONSLEVEL1
    1563: 15,  # RESEARCH_PROTOSSAIRWEAPONSLEVEL2
    1564: 16,  # RESEARCH_PROTOSSAIRWEAPONSLEVEL3
    1065: 4,  # RESEARCH_PROTOSSGROUNDARMORLEVEL1
    1066: 5,  # RESEARCH_PROTOSSGROUNDARMORLEVEL2
    1067: 6,  # RESEARCH_PROTOSSGROUNDARMORLEVEL3
    1062: 1,  # RESEARCH_PROTOSSGROUNDWEAPONSLEVEL1
    1063: 2,  # RESEARCH_PROTOSSGROUNDWEAPONSLEVEL2
    1064: 3,  # RESEARCH_PROTOSSGROUNDWEAPONSLEVEL3
    1068: 7,  # RESEARCH_PROTOSSSHIELDSLEVEL1
    1069: 8,  # RESEARCH_PROTOSSSHIELDSLEVEL2
    1070: 9,  # RESEARCH_PROTOSSSHIELDSLEVEL3
    1126: 13,  # RESEARCH_PSISTORM
    2720: 25,  # RESEARCH_SHADOWSTRIKE
    1568: 20,  # RESEARCH_WARPGATE

    # Train
    922: 3,   # TRAIN_ADEPT
    948: 17,   # TRAIN_CARRIER
    978: 8,   # TRAIN_COLOSSUS
    920: 6,   # TRAIN_DARKTEMPLAR
    994: 9,   # TRAIN_DISRUPTOR
    919: 5,   # TRAIN_HIGHTEMPLAR
    979: 11,   # TRAIN_IMMORTAL
    110: 19,   # TRAIN_MOTHERSHIP
    1853: 20,  # TRAIN_MOTHERSHIPCORE
    977: 10,   # TRAIN_OBSERVER
    954: 14,   # TRAIN_ORACLE
    946: 13,   # TRAIN_PHOENIX
    1006: 0,  # TRAIN_PROBE
    921: 2,   # TRAIN_SENTRY
    917: 4,   # TRAIN_STALKER
    955: 16,   # TRAIN_TEMPEST
    950: 15,   # TRAIN_VOIDRAY
    976: 12,   # TRAIN_WARPPRISM
    916: 1,   # TRAIN_ZEALOT

    # TrainWarp
    1419: 3,  # TRAINWARP_ADEPT
    1417: 6,  # TRAINWARP_DARKTEMPLAR
    1416: 5,  # TRAINWARP_HIGHTEMPLAR
    1418: 2,  # TRAINWARP_SENTRY
    1414: 4,  # TRAINWARP_STALKER
    1413: 1,  # TRAINWARP_ZEALOT

    # # Cancel
    # 3659,  # CANCEL
    # 313,   # CANCELSLOT_ADDON
    # 305,   # CANCELSLOT_QUEUE1
    # 307,   # CANCELSLOT_QUEUE5
    # 309,   # CANCELSLOT_QUEUECANCELTOSELECTION
    # 1832,  # CANCELSLOT_QUEUEPASSIVE
    # 314,   # CANCEL_BUILDINPROGRESS
    # 3671,  # CANCEL_LAST
    # 1848,  # CANCEL_MORPHMOTHERSHIP
    # 304,   # CANCEL_QUEUE1
    # 306,   # CANCEL_QUEUE5
    # 312,   # CANCEL_QUEUEADDON
    # 308,   # CANCEL_QUEUECANCELTOSELECTION
    # 1831,  # CANCEL_QUEUEPASIVE
    # 1833,  # CANCEL_QUEUEPASSIVECANCELTOSELECTION

    # # Stop
    # 3665,  # STOP
    # 2057,  # STOP_BUILDING
    # 4,     # STOP_STOP
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
                            print('                                                    counter value: ' + str(counter.value))
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

                            time_step = 1

                            try:
                                while True:
                                    time_step += step_mul
                                    controller.step(step_mul)
                                    obs = controller.observe()

                                    new_data_points = []

                                    for action in obs.actions:
                                        if is_macro_action(action):
                                            resources = get_resources(obs.observation)
                                            upgrades = get_upgrades(obs.observation.raw_data.player.upgrade_ids)
                                            in_progress = get_units_in_progress(obs.observation.raw_data.units)
                                            friendly_unit_list = get_friendly_unit_list(obs.observation.raw_data.units)
                                            enemy_unit_list = get_enemy_unit_list(obs.observation.raw_data.units)

                                            new_data_point = []
                                            new_data_point.append(time_step)
                                            new_data_point += resources
                                            new_data_point += upgrades
                                            new_data_point += in_progress
                                            new_data_point += friendly_unit_list
                                            new_data_point += enemy_unit_list
                                            new_data_point.append(action.action_raw.unit_command.ability_id)

                                            new_data_points.append(new_data_point)

                                    data_points = data_points + new_data_points

                                    # The game has finished if there is a player_result
                                    if obs.player_result:
                                        print('                                                    Replay #' + str(index+1) + ' from player ' + str(player.player_info.player_id) + '\'s perspective has finished.')
                                        break
                            except KeyboardInterrupt:
                                return
                        
                        print('                                                    Finished processing game #' + str(index + 1) + '. Saving data...')
                        save_replay_data(replay_save_path, data_points)

                    # Everything went smothely with this replay batch. Reset the counter and move on.
                    replays_range_left_in_batch = (0,0)

        except KeyboardInterrupt:
            return
        except WebSocketTimeoutException:
            print('                                                    Websocket timed out.')
        except pysc2.lib.protocol.ConnectionError:
            print('                                                    Websocket timed out.')
        except:
            print('                                                    Something want wrong. Skipping this replay.')

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
                protoss_unit = protoss_unit_mapper.get(unit.unit_type)
                in_progress_list[protoss_unit] += 1

            # Something is being built/something by something else. E.g. a building training a unit.
            if unit.orders is not None:
                for order in unit.orders:
                    # The unit being built by e.g. a building.
                    in_progress_entity = protoss_action_to_unit_mapper.get(order.ability_id) 
        
                    if in_progress_entity is not None:
                        in_progress_list[in_progress_entity] += 1

    return in_progress_list


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
        if unit.alliance == 4:
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


def is_macro_action(action):
    if (hasattr(action, "action_raw") 
        and hasattr(action.action_raw, "unit_command")
        and protoss_action_to_unit_mapper.get(action.action_raw.unit_command.ability_id) is not None
        ):
            return True
    
    return False


def save_replay_data(save_path, replay_data):
    np.save(save_path, replay_data)


def main(argv):
    jobs = []                   # The list of processes
    counter = Value('i', FLAGS.start_from_replay)     # Lock so the processes don't work on the same replays

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
    # counter = Value('i', 0)     # Lock so the processes don't work on the same replays
    # extract_actions(counter, FLAGS.replays_path, FLAGS.save_path, FLAGS.batch_size, FLAGS.step_mul)

    app.run(main)