import argparse
import json
from scipy import signal
import matplotlib.pyplot as plt


def visualize_log(filename, figsize=None, output=None,label = None):
    if label is None:
        label = 'epsilon 0.'+filename[10]
    with open(filename, 'r') as f:
        data = json.load(f)
    if 'episode' not in data:
        raise ValueError('Log file "{}" does not contain the "episode" key.'.format(filename))
    episodes = data['episode']

    # Get value keys. The x axis is shared and is the number of episodes.
    keys = sorted(list(set(data.keys()).difference(set(['episode']))))
    line = plt.plot(episodes,signal.savgol_filter(data["episode_reward"],91,2),label = label)
    return line


parser = argparse.ArgumentParser()
filename = "dqn_pacman_log.json"
filename = "dqn_pacman_exp_log.json"
# filename = "dqn_pacman_cons_log.json"
filename = "dqn_pacman_lin_log.json"
parser.add_argument('--output', type=str, default=None, help='The output file. If not specified, the log will only be displayed.')
parser.add_argument('--figsize', nargs=2, type=float, default=None, help='The size of the figure in `width height` format specified in points.')
args = parser.parse_args()

plt.ylabel("episode_reward")
plt.xlabel('episodes')
plt.tight_layout()


# filename = "dqn_pacman_lin_log.json"
# visualize_log(filename, output=args.output, figsize=args.figsize)
# filename = "dqn_pacman_exp_log.json"
# visualize_log(filename, output=args.output, figsize=args.figsize)
# filename = "dqn_pacman_cons_log.json"
# visualize_log(filename, output=args.output, figsize=args.figsize)

filename = "dqn_pacman{}_cons_log.json"
lines = []
# for i in range(1,10):
#     line = visualize_log(filename.format(str(i)), output=args.output, figsize=args.figsize)
#     lines.append(line[0])

# filename = "dqn_pacman{}_long_cons_log.json"
# for i in [1,9]:
#     line = visualize_log(filename.format(str(i)), output=args.output, figsize=args.figsize)
#     lines.append(line[0])

l = ["constant","linear","exponential"]
filenames = ["dqn_pacman_constant_long_log.json","dqn_pacman_linear_long_log.json","dqn_pacman_exponential_long_log.json"]
for i in range(len(filenames)):
    line = visualize_log(filenames[i], output=None, figsize=2,label = l[i])
    lines.append(line[0])

plt.legend(handles = lines)
# filename = "dqn_pacman1_long_cons_log.json"
# visualize_log(filename, output=args.output, figsize=args.figsize)

plt.show()