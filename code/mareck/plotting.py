import matplotlib
import matplotlib.pyplot as plt

def plot(axes, title, data, ticks=None, tlabels=None) :
    ts, ys, labels, colors = data
    axes.set_title(title)
    axes.grid(True)
    for t, y, label, color in zip(ts, ys, labels, colors) :
        axes.plot(t, y, label=label, color=color)
    lst = list(set(labels))
    if not (len(lst) == 1 and lst[0] is None) :
        axes.legend()
    if ticks is not None and tlabels is not None :
        axes.set_xticks(ticks)
        axes.set_xticklabels(tlabels)

def bar(axes, title, data, ticks=None, tlabels=None) :
    xss, xms, ths, colors, zs = data
    axes.set_title(title)
    axes.grid(zorder=0)
    for xs, xm, th, color, z in zip(xss, xms, ths, colors, zs) :
        axes.bar(xs, xm, th, color=color, zorder=z)
    if ticks is not None and tlabels is not None :
        axes.set_xticks(ticks)
        axes.set_xticklabels(tlabels)
