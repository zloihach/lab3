from matplotlib import rc
import regress as reg

if __name__ == "__main__":

    font = {'family': 'Verdana', 'weight': 'normal'}
    rc('font', **font)
    reg.oneRegress()
    reg.twoRegress()
