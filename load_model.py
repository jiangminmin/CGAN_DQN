import tensorflow as tf
import random
import numpy as np
from sample_env_offline import freq_env
import matplotlib.pylab as plt
def waterfall_reset():
	start_freq = 97000000.0
	end_freq = 98000000.0
	freq_line = []
	power_ls = []
	user_freq = [6.94817642133, 11.0710678659, 14.6195817555, 17.3627686085,
				 22.8806895204, 24.2871283707, 24.7753483924, 25.2412643153,
				 25.6009231012, 25.3235259942, 24.7298037759, 23.9569713324,
				 23.1641902552, 19.5581498392, 15.0973001711, 11.5414549903]
	with open('data97_98_97.4.txt') as fr:
		for line in fr.readlines():
			curLine = line.strip().split()
			freq_line.append(curLine[2])
			power_ls.append(curLine[3])
	index_end = [i for i, x in enumerate(freq_line) if x == str(end_freq)]
	rand_int = random.randint(0, 9)
	other_1 = random.randint(0, 9)
	other_2 = random.randint(0, 9)
	waterfall_reset = np.zeros(shape=(1, 100, 160, 1), dtype=float)
	for x in range(0, int(index_end[0])-60):
		for y in range(0, int(index_end[0])):
			waterfall_reset[0, x, y] = float(power_ls[(int(index_end[0]) + 1) * (x + 1) + y])
	for m in range(0, int(index_end[0])-60):
		for n in range(rand_int * 16, (rand_int + 1) * 16):
			waterfall_reset[0, 0:-4, n] = waterfall_reset[0, m, n, 0] + user_freq[n - rand_int * 16]
	for i in range(len(user_freq)):
		waterfall_reset[0, -4, i + 16 * 3] = waterfall_reset[0, -4, i + 16 * 3, 0] + user_freq[i]
		waterfall_reset[0, -3, i + 16 * 4] = waterfall_reset[0, -3, i + 16 * 4, 0] + user_freq[i]
		waterfall_reset[0, -2, i+16*3] = waterfall_reset[0, -2, i+16*4, 0] + user_freq[i]
		waterfall_reset[0, -1, i+16*3] = waterfall_reset[0, -1, i+16*4,0] + user_freq[i]
	return waterfall_reset
def load_model():
    with tf.Session() as sess:
		env = freq_env()
		waterfall = env.waterfall_reset()
		plt.imshow(waterfall[0, :, :, 0])
		plt.show()
		waterfall_re = np.reshape(waterfall,[8000])
		#saver = tf.train.Saver()
		#path = "./dqn"
		#ckpt = tf.train.get_checkpoint_state(path)
		#saver.restore(sess, ckpt.model_checkpoint_path)
		saver = tf.train.import_meta_graph('./dqn/model-1000.cptk.meta')
		tf.reset_default_graph()
		saver.restore(sess,tf.train.latest_checkpoint(""))
		graph = tf.get_default_graph()
		y = tf.get_collection('pred_network')[0]
		#input_x = graph.get_tensor_by_name('s:0')
		#y = graph.get_operation_by_name('q_eval').outputs[0]
		#input_x = graph.get_operation_by_name('s').outputs[0]
		print(np.argmax(sess.run(y,feed_dict={"s:0":waterfall_re})))

if __name__ == "__main__":
	load_model()