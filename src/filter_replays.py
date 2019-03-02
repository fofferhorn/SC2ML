import os
from multiprocessing import Process, Value
from shutil import copyfile

from absl import app
from absl import flags
import sys

from pysc2 import run_configs
from s2clientprotocol import common_pb2 as sc_common

FLAGS = flags.FLAGS

# General settings
flags.DEFINE_string(name = 'replays_path', default = 'replays', help = 'The path to the replays to filter from the current directory.')
flags.DEFINE_string(name = 'save_path', default = 'filtered_replays', help = 'The path to the folder to save the replays in from the current directory.')
flags.DEFINE_integer(name = 'n_instance', default = 8, help = 'The default amount of threads to use to filter the replays.')
flags.DEFINE_integer(name = 'batch_size', default = 10, help = 'The amount of replays each worker process takes at a time.')

# Flags for filtering replays
flags.DEFINE_integer(name = 'min_duration', default = 1350, help = 'The minimum duration of the game in game ticks.')
flags.DEFINE_integer(name = 'max_duration', default = None, help = 'The maximum duration of the game in game ticks.')
flags.DEFINE_integer(name = 'min_apm', default = 10, help = 'The minimum APM of both players of the game.')
flags.DEFINE_integer(name = 'max_apm', default = None, help = 'The maximum APM of both players of the game.')
flags.DEFINE_integer(name = 'min_mmr', default = 1000, help = 'The minimum MMR of both players of the game.')
flags.DEFINE_integer(name = 'max_mmr', default = None, help = 'The maximum MMR of both players of the game.')

FLAGS(sys.argv)


def valid_replay(info, ping):

    if info.HasField("error"):
        return False
    if info.game_duration_loops < FLAGS.min_duration:
        return False
    if FLAGS.max_duration is not None and info.game_duration_loops > FLAGS.max_duration:
        return  False
    if len(info.player_info) != 2:
        return False

    for p in info.player_info:
        if sc_common.Race.Name(p.player_info.race_actual) != 'Protoss':
            return False
        if p.player_apm < FLAGS.min_apm:
            # Low MMR = corrupt replay or player who is weak.
            return False
        if FLAGS.max_apm is not None and p.player_apm > FLAGS.max_apm:
            return False
        if FLAGS.min_mmr is not None and p.player_mmr < FLAGS.min_mmr:
            # Low APM = player just standing around.
            return False
        if FLAGS.max_mmr is not None and p.player_mmr > FLAGS.max_mmr:
            return False
        if p.player_result.result not in {1, 2}:
            return False

    return True


def filter_replays(counter, replays_path, save_path, batch_size, run_config):
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
            # Check if we have filtered all the replays
            with counter.get_lock():
                if counter.value * batch_size > len(replay_paths):
                    break
                i = counter.value
                counter.value += 1

            batch_start = i * batch_size
            batch_end = i * batch_size + batch_size

            if batch_end > len(replay_paths) - 1:
                batch_end = len(replay_paths) - 1
            
            for index in range(batch_start, batch_end):
                print('================================================================================ Processing replay #' + str(index + 1))
                sys.stdout.flush()

                replay_path = replay_paths[index]

                replay_data = run_config.replay_data(replay_path)
                
                info = controller.replay_info(replay_data)

                info = controller.replay_info(replay_data)
                ping = ping = controller.ping()

                if valid_replay(info, ping):
                    print('================================================================================ Found valid game #' + str(index + 1))

                    replay_name = replay_path.split('\\')

                    replay_save_path = os.path.join(cwd, save_path, replay_name[-1])
                    copyfile(replay_path, replay_save_path)



def main(argv):
    jobs = []                   # The list of processes
    counter = Value('i', 0)     # Lock so the processes don't work on the same replays
    run_config = run_configs.get()

    for i in range(FLAGS.n_instance):
        p = Process(target = filter_replays, args = [
            counter, 
            FLAGS.replays_path, 
            FLAGS.save_path, 
            FLAGS.batch_size,
            run_config])
        jobs.append(p)
        p.start()
    
    # Wait for each process to finish.
    for i in range(FLAGS.n_instance):
        jobs[i].join()
    

if __name__ == "__main__":
    
    counter = Value('i', 0)
    run_config = run_configs.get()

    filter_replays(counter, 
            FLAGS.replays_path, 
            FLAGS.save_path, 
            FLAGS.batch_size,
            run_config)
    
    #app.run(main)