import tensorflow as tf
import numpy as np
from skimage import transform
import matplotlib.pylab as plt
from model.gan_load_model import generator
import cv2


def align_data(data):
    """Given dict with lists, creates aligned strings

    Adapted from Assignment 3 of CS224N

    Args:
        data: (dict) data["x"] = ["I", "love", "you"]
              (dict) data["y"] = ["O", "O", "O"]

    Returns:
        data_aligned: (dict) data_align["x"] = "I love you"
                           data_align["y"] = "O O    O  "

    """
    data_aligned = dict()

    # for each entry, create aligned string
    for key, seq in data.items():
        data_aligned[key] = seq

    return data_aligned


def get_model_api():
    model = generator()
    in_out_collections = model.create_model()
    saver = tf.train.Saver()
    checkpoint = tf.train.latest_checkpoint("cgan_weights")
    sess = tf.Session()
    print("loading model from checkpoint")
    saver.restore(sess, checkpoint)
    def model_api(input_data,):
        preds = sess.run(in_out_collections.outputs, feed_dict={in_out_collections.my_inputs:input_data})
        preds = np.reshape(preds, (256, 256))
        preds = transform.resize(preds, (100, 1601), mode='constant')
        preds = preds * 255
        return preds
    return model_api

"""
if __name__ == "__main__":
    sweep = np.zeros(shape=(100, 1601))
    row = 0
    last_interval=5
    interval=5
    action_space = np.linspace(80, 1520, 19, dtype=np.int)
    n_actions = len(action_space)
    for i in range(100):
        if i - last_interval == 0:
            # self.interval = np.random.randint(2,11)
            last_interval += interval
            row += 1
        randint = np.random.randint(0, 40)
        sweep[i, action_space[row % n_actions] - 10 -
                 randint:action_space[row % n_actions] + 11 + randint] = 1
    sweep[:, 84:109] += 2
    sweep[:, 152:176] += 2
    sweep[:, 279:301] += 2
    sweep[:, 390:411] += 2
    sweep[:, 615:640] += 2
    sweep[:, 944:954] += 2
    sweep[:, 960:992] += 2
    sweep[:, 1000:1009] += 2
    sweep[:, 1045:1066] += 2
    sweep[:, 1157:1180] += 2
    sweep = np.array(sweep[0:100, :], )
    sweep = sweep * 255 / 3
    #cv2.imwrite("test.jpg",sweep)
    #label = cv2.imread("test.jpg",0)

    model_api = get_model_api()
    output_data = model_api("test.jpg")
    plt.imshow(output_data, interpolation='nearest', aspect='auto')
    plt.show()
"""