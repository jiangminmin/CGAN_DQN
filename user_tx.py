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
import globalvar as globalvar
import numpy as np
class user(grc_wxgui.top_block_gui):
    def __init__(self):
        grc_wxgui.top_block_gui.__init__(self, title="Mc")

        ##################################################
        # Variables
        ##################################################
        self.variable_0 = variable_0 = 0
        self.samp_rate_0_0 = samp_rate_0_0 = 32000
        self.samp_rate_0 = samp_rate_0 = 32000
        self.samp_rate = samp_rate = 32000
        self.center_freq = center_freq = 99000000.0

        ##################################################
        # Blocks
        ##################################################
        #self.uhd_usrp_sink_0 = uhd.usrp_sink(
        #	",".join(("", "")),
        #	uhd.stream_args(
        #		cpu_format="fc32",
        #		channels=range(1),
        #	),
        #)
        self.uhd_usrp_sink_0 = uhd.usrp_sink(device_addr="serial=30FA179",stream_args=uhd.stream_args(cpu_format="fc32",channels=range(1)))
        self.uhd_usrp_sink_0.set_samp_rate(88200)
        self.uhd_usrp_sink_0.set_center_freq(center_freq, 0)
        self.uhd_usrp_sink_0.set_gain(30, 0)
        self.uhd_usrp_sink_0.set_antenna('TX/RX', 0)
        self.uhd_usrp_sink_0.set_bandwidth(250000, 0)
        self.blocks_wavfile_source_0 = blocks.wavfile_source('/home/jiangminmin/Downloads/haha.wav', True)
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
user_obj = user()
def main_user_tx(user_obj=user()):
    #user = top_block_cls()
    user_obj.Start(True)
    user_obj.wait()
def set_user_freq(user_freq,user_obj=user()):
    user_obj.set_center_freq(user_freq)
    print('user_freq change to:',user_obj.get_center_freq())
def main():
    thread_1=threading.Thread(target=main_user_tx)
    thread_1.start()

if __name__ == '__main__':
    main()
    set_user_freq(1910000000.0)
"""