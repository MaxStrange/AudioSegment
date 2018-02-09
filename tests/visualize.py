"""
This file provides a function for visualizing an audiosegment
in the time domain.
"""
import audiosegment as asg
import matplotlib.pyplot as plt

def visualize(seg, title=""):
    plt.plot(seg.to_numpy_array())
    plt.title(title)
    plt.show()
    plt.clf()

