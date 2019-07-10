#coding=utf-8
from numpy import arange,sin,pi
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
import time
from tkinter import *
root = tk.Tk()
root.title("matplotlib in TK")
f = Figure(figsize=(6,6),dpi=100)
canvas = FigureCanvasTkAgg(f,master=root)
canvas.get_tk_widget().pack(side=tk.TOP,fill=tk.BOTH,expand=1)

t = arange(0.0,3,0.01)
s = sin(2*pi*t)
"""
for i in range(4):
    f.clf()
    a = f.add_subplot(111)
    a.plot(t,s)
    canvas.draw()
    time.sleep(1)
#
pass
"""
plt.subplot(121)
for i in range(0,100,10):
    plt.axvline(i,color='red')
plt.plot(s)
plt.subplot(122)
plt.plot(s)
plt.show()
