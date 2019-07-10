from gnuradio import gr, eng_notation
from gnuradio import uhd
from gnuradio import analog
from gnuradio import blocks
from gnuradio.eng_option import eng_option
from optparse import OptionParser
import math
import sys
from grc_gnuradio import wxgui as grc_wxgui
from gnuradio.wxgui import stdgui2, fftsink2
import wx
import numpy as np
import time
import threading

class pipeline():
    def __init__(self, filename, lo_freq, audio_rate, if_rate):


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
        self.connect((self.analog_wfm_tx_0, 0), self)
        self.connect((self.blocks_wavfile_source_0, 0), (self.analog_wfm_tx_0, 0))

class fm_tx_block(stdgui2.std_top_block):
    def __init__(self, frame, panel, vbox, argv):
        MAX_CHANNELS = 7
        #grc_wxgui.top_block_gui.__init__(self, title="Mc")
        stdgui2.std_top_block.__init__ (self, frame, panel, vbox, argv)

        parser = OptionParser (option_class=eng_option)
        parser.add_option("-a", "--args", type="string", default="",
                          help="UHD device address args [default=%default]")
        parser.add_option("", "--spec", type="string", default=None,
	                  help="Subdevice of UHD device where appropriate")
        parser.add_option("-A", "--antenna", type="string", default='RX2',
                          help="select Rx Antenna where appropriate")
        parser.add_option("-s", "--samp-rate", type="eng_float", default=96000,
                          help="set sample rate (bandwidth) [default=%default]")
        parser.add_option("-f", "--freq", type="eng_float", default=100e6,
                          help="set frequency to FREQ", metavar="FREQ")
        parser.add_option("-g", "--gain", type="eng_float", default=40,
                          help="set gain in dB (default is midpoint)")
        parser.add_option("-n", "--nchannels", type="int", default=1,
                           help="number of Tx channels [1,4]")
        #parser.add_option("","--debug", action="store_true", default=False,
        #                  help="Launch Tx debugger")
        (options, args) = parser.parse_args ()

        if len(args) != 0:
            parser.print_help()
            sys.exit(1)

        if options.nchannels < 1 or options.nchannels > MAX_CHANNELS:
            sys.stderr.write ("fm_tx4: nchannels out of range.  Must be in [1,%d]\n" % MAX_CHANNELS)
            sys.exit(1)

        # ----------------------------------------------------------------
        # Set up constants and parameters

        self.u = uhd.usrp_sink(device_addr="serial=30FA185", stream_args=uhd.stream_args('fc32'))
        self.u.set_bandwidth(5000000, 0)
        # Set the subdevice spec
        if(options.spec):
            self.u.set_subdev_spec(options.spec, 0)

        # Set the antenna
        if(options.antenna):
            self.u.set_antenna(options.antenna, 0)

        self.usrp_rate = options.samp_rate
        self.u.set_samp_rate(self.usrp_rate)
        self.usrp_rate = self.u.get_samp_rate()

        self.sw_interp = 10
        self.audio_rate = self.usrp_rate / self.sw_interp    # 32 kS/s

        if options.gain is None:
            # if no gain was specified, use the mid-point in dB
            g = self.u.get_gain_range()
            options.gain = float(g.start()+g.stop())/2

        self.set_gain(options.gain)
        self.set_freq(options.freq)

        self.blocks_wavfile_source_0 = blocks.wavfile_source('/home/jiangminmin/Downloads/haha.wav', True)
        self.analog_wfm_tx_0 = analog.wfm_tx(
            audio_rate=44100,
            quad_rate=88200,
            tau=75e-6,
            max_dev=75e3,
            fh=-1.0,
        )
        #self.connect((self.analog_wfm_tx_0, 0), self)
        self.connect((self.blocks_wavfile_source_0, 0), (self.analog_wfm_tx_0, 0))
        #t = pipeline('/home/jojo/Downloads/haha.wav', 0,self.audio_rate, self.usrp_rate)
        self.connect ((self.analog_wfm_tx_0, 0), self.u)

    def set_freq(self, target_freq):
        r = self.u.set_center_freq(target_freq, 0)
        if r:
            print "Frequency =", eng_notation.num_to_str(self.u.get_center_freq())
            return True
        return False

    def set_gain(self, gain):
        self.u.set_gain(gain, 0)

def main():
    freq_ls = [i for i in np.arange(100500000.0, 110000000.0, 500000)]
    user = fm_tx_block(0,0,0,0)
    user.start()
    while 1:
        for freq_index in xrange(len(freq_ls)):
            user.set_freq(freq_ls[freq_index])
            time.sleep(0.5)
    #user.wait()


if __name__ == '__main__':
    main ()
