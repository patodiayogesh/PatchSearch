
from stats_experiments import filenames
from statsmodels.stats.weightstats import ztest as ztest

def dist_diff(file1, file2, test):

    with open(file1,'r') as f:
        dist_1 = [float(line.strip('\n')) for line in f]
    with open(file2, 'r') as f:
        dist_2 = [float(line.strip('\n')) for line in f]

    if test == 'z-test':
        values = ztest(dist_1, dist_2)
        print(values)

if __name__=='__main__':
    for file_pairs in filenames:
        dist_diff(file_pairs[0], file_pairs[1], 'z-test')


