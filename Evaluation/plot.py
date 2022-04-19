from plot_experiments import filenames
import matplotlib.pyplot as plt

def plot_edit_dist_(files):

    file_1 = files[0]
    file_2 = files[1]
    x, y = [], []
    with open(file_1) as f:
        line = float(f.readline().strip('\n'))
        y.append(line)
    with open(file_1) as f:
        line = float(f.readline().strip('\n'))
        x.append(line)

    data_pair = zip(y, x)

    plt.scatter(*zip(*data_pair))
    plt.xlabel('Prev Code Normalized Edit Distance')
    plt.ylabel('Buggy Code Normalized Edit Distance')
    plt.title('Buggy Code vs Prev Code')
    plt.savefig(file_1+'_'+file_2)
    plt.clf()

for pair in filenames:
    plot_edit_dist_(pair)