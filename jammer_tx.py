#!/usr/bin/env python2
# -*- coding: utf-8 -*-
##################################################
# GNU Radio Python Flow Graph
# Title: Mc
# Generated: Tue Oct 16 10:29:03 2018
##################################################

if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print "Warning: failed to XInitThreads()"

from gnuradio import analog
from gnuradio import blocks
from gnuradio import eng_notation
from gnuradio import gr
from gnuradio import uhd
from gnuradio.eng_option import eng_option
from gnuradio.filter import firdes
from grc_gnuradio import wxgui as grc_wxgui
from optparse import OptionParser
import threading
import time
import wx
import random
import numpy as np
class jammer(grc_wxgui.top_block_gui):
    def __init__(self):
        grc_wxgui.top_block_gui.__init__(self, title="Mc")

        ##################################################
        # Variables
        ##################################################
        self.variable_0 = variable_0 = 0
        self.samp_rate_0_0 = samp_rate_0_0 = 32000
        self.samp_rate_0 = samp_rate_0 = 32000
        self.samp_rate = samp_rate = 32000
        self.center_freq = center_freq = 10000000.0

        ##################################################
        # Blocks
        ##################################################

        #self.uhd_usrp_sink_0 = uhd.usrp_sink(
        #	",".join(("", "")),
        #    uhd_addr="serial=30FA181",
        #	uhd.stream_args(
        #		cpu_format="fc32",
        #		channels=range(1),
        #	)
        #)

        self.uhd_usrp_sink_0 = uhd.usrp_sink(device_addr="serial=30FA214",stream_args=uhd.stream_args(cpu_format="fc32",channels=range(1)))

        self.uhd_usrp_sink_0.set_samp_rate(88200)
        self.uhd_usrp_sink_0.set_center_freq(center_freq, 0)
        self.uhd_usrp_sink_0.set_gain(30, 0)
        self.uhd_usrp_sink_0.set_antenna('TX/RX', 0)
        self.uhd_usrp_sink_0.set_bandwidth(250000, 0)#250000
        self.blocks_wavfile_source_0 = blocks.wavfile_source('/home/jmm/Downloads/Wind_Shakes_the_Barley.wav', True)
        self.analog_wfm_tx_0 = analog.wfm_tx(
        	audio_rate=44100,
        	quad_rate=88200,
        	tau=75e-6,
        	max_dev=75e3,
        	fh=-1.0,
        )

        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_wfm_tx_0, 0), (self.uhd_usrp_sink_0, 0))
        self.connect((self.blocks_wavfile_source_0, 0), (self.analog_wfm_tx_0, 0))

    def get_variable_0(self):
        return self.variable_0

    def set_variable_0(self, variable_0):
        self.variable_0 = variable_0

    def get_samp_rate_0_0(self):
        return self.samp_rate_0_0

    def set_samp_rate_0_0(self, samp_rate_0_0):
        self.samp_rate_0_0 = samp_rate_0_0

    def get_samp_rate_0(self):
        return self.samp_rate_0

    def set_samp_rate_0(self, samp_rate_0):
        self.samp_rate_0 = samp_rate_0

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate

    def get_center_freq(self):
        return self.center_freq

    def set_center_freq(self, center_freq):
        self.center_freq = center_freq
        self.uhd_usrp_sink_0.set_center_freq(center_freq, 0)
"""
i = 0 
jam_freq = 0
def main(top_block_cls=mc, options=None): 
    global i
    list_freq = []
    for freq_point in range(100000000,110000001,1000000):
	   list_freq.append(freq_point)
    #rand_freq = random.sample(list_freq,1)  
    #jammer = top_block_cls(rand_freq[0])
    jammer = top_block_cls(list_freq[i])
    #jammer.set_center_freq(list_freq[i])
    globalvar._init()
    globalvar.set_value('jammer_freq',jammer.get_center_freq())
    print(jammer.get_center_freq())
    jammer.Start(True)
    time.sleep(2)
    jammer.stop()
    if i == list_freq.index(list_freq[-1]):
        i = 0
    else:
        i += 1
    timer = threading.Timer(0.5,main)
    timer.start()

        
"""
"""
#jammer_obj=jammer()
def main_jammer_tx(jammer_obj=jammer()):
    freq_ls = np.linspace(100500000, 109500000, 19)
    #rand_freq = random.sample(list_freq,1)
    #jammer = top_block_cls(rand_freq[0])
    jammer_obj.Start(True)
    while 1:
        for freq_index in xrange(len(freq_ls)):
            jammer_obj.set_center_freq(freq_ls[np.random.randint(0,19)])
            time.sleep(np.random.randint(1,6))
            print('jammer_freq:',jammer_obj.get_center_freq())
        #jammer_obj.wait()

def ret_freq(jammer_obj):
    freq_ls = [i for i in np.arange(129000000.0, 130000001.0, 100000)]
    while 1:
        for freq_index in range(len(freq_ls)):
            jammer_obj.set_center_freq(freq_ls[freq_index])
            time.sleep(2)

def get_jammer_freq(jammer_obj=jammer()):
    #thread_jammer = threading.Thread(target=jammer.get_center_freq)
    #thread_jammer.start()
    #thread_jammer.join()
    return jammer_obj.get_center_freq()

def main():
    threads_main=threading.Thread(target=main_jammer_tx)
    threads_main.start()

if __name__ == '__main__':
    main_jammer_tx()
"""