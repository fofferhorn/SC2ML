import os
from multiprocessing import Process, Value
from shutil import copyfile

from absl import app
from absl import flags
import sys

from pysc2 import run_configs
from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import sc2api_pb2 as sc_pb

FLAGS = flags.FLAGS

# General settings
flags.DEFINE_string(name = 'replays_path', default = 'filtered_replays', help = 'The path to the replays to extract actions from the current directory.')
flags.DEFINE_string(name = 'save_path', default = 'extracted_actions', help = 'The path to the folder to save the replays in from the current directory.')
flags.DEFINE_integer(name = 'n_instance', default = 8, help = 'The default amount of threads to use to filter the replays.')
flags.DEFINE_integer(name = 'batch_size', default = 10, help = 'The amount of replays each worker process takes at a time.')
flags.DEFINE_integer(name = 'step_mul', default = 1, help = 'The amount of game steps between each observation.')


FLAGS(sys.argv)


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

    with run_config.start() as controller:
        while True:
            with counter.get_lock():
                if counter.value * batch_size > len(replays_path):
                    break
                i = counter.value
                counter.value += 1

            batch_start = i * batch_size
            batch_end = i * batch_size + batch_size

            if batch_end > len(replay_paths) - 1:
                batch_end = len(replay_paths) - 1
            
            for index in range(batch_start, batch_end):
                print('================================================================================ Processing replay #' + str(index + 1))

                replay_path = replay_paths[index]

                replay_data = run_config.replay_data(replay_path)
                info = controller.replay_info(replay_data)

                map_data = None
                if info.local_map_path:
                    map_data = run_config.map_data(info.local_map_path)

                interface = sc_pb.InterfaceOptions(raw=True, score=False,
                                                feature_layer=sc_pb.SpatialCameraSetup(width=24))
                
                for player in info.player_info:

                    start_replay = sc_pb.RequestStartReplay(
                        replay_data = replay_data,
                        options = interface,
                        disable_fog = FLAGS.disable_fog,
                        observed_player_id = player.player_info.player_id)

                    actions = []

                    controller.step()

                    try:
                        while True:
                            controller.step(FLAGS.step_mul)
                            obs = controller.observe()

                            print(obs)

                            exit(0)

                    except KeyboardInterrupt:
                        pass

        

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