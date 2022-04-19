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

    plt.scatter(x, y, c=['r','g'])
    plt.set_xlabel('Prev Code Normalized Edit Distance')
    plt.set_ylabel('Buggy Code Normalized Edit Distance')
    plt.title('Buggy Code vs Prev Code')

for pair in filenames:
    plot_edit_dist_(pair)