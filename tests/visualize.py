"""
This file provides a function for visualizing an audiosegment
in the time domain.
"""
import audiosegment as asg
import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt

VIS_MS = 3000 # The number of ms to use in visualizing stuff

def visualize(seg, title=""):
    plt.plot(seg.to_numpy_array())
    plt.title(title)
    plt.show()
    plt.clf()

