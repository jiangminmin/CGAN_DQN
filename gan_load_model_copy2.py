import tensorflow as tf
import numpy as np
import collections
import cv2
import matplotlib.pylab as plt
from skimage import transform,data
#EPS = 1e-12
#CROP_SIZE = 256

Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")
#Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train")
class generator():
    def __init__(self):
        self.EPS = 1e-12
        self.CROP_SIZE = 256
        self.Model = collections.namedtuple("Model","my_inputs, outputs")# predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train")
        self.action_space = np.linspace(80, 1520, 19, dtype=np.int)
        self.n_actions = len(self.action_space)
        self.lines = 100
        self.currentline = 0
        self.interval = 5
        self.last_interval = 5
        #self.img =self.create_sweep()
        #self.interference = self.generate_img(self.img)
    def discrim_conv(self,batch_input, out_channels, stride):
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        return tf.layers.conv2d(padded_input, out_channels, kernel_size=4, strides=(stride, stride), padding="valid", kernel_initializer=tf.random_normal_initializer(0, 0.02))
    def lrelu(self,x, a):
        with tf.name_scope("lrelu"):
            x = tf.identity(x)
            return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)
    def gen_conv(self,batch_input, out_channels):
        # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
        initializer = tf.random_normal_initializer(0, 0.02)
        return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)
    def gen_deconv(self,batch_input, out_channels):
        # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
        initializer = tf.random_normal_initializer(0, 0.02)
        return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)
    def preprocess(self,image):
        with tf.name_scope("preprocess"):
            # [0, 1] => [-1, 1]
            return image * 2 - 1
    def deprocess(self,image):
        with tf.name_scope("deprocess"):
            # [-1, 1] => [0, 1]
            return (image + 1) / 2
    def batchnorm(self,inputs):
        return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))
    def create_generator(self,generator_inputs, generator_outputs_channels):
        layers = []

        # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
        with tf.variable_scope("encoder_1"):
            output = self.gen_conv(generator_inputs, 64)
            layers.append(output)

        layer_specs = [
            64 * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
            64 * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
            64 * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
            64 * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
            64 * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
            64 * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
            64 * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
        ]

        for out_channels in layer_specs:
            with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
                rectified = self.lrelu(layers[-1], 0.2)
                # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                convolved = self.gen_conv(rectified, out_channels)
                output = self.batchnorm(convolved)
                layers.append(output)

        layer_specs = [
            (64 * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
            (64 * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
            (64 * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
            (64 * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
            (64 * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
            (64 * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
            (64 , 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
        ]

        num_encoder_layers = len(layers)
        for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
            skip_layer = num_encoder_layers - decoder_layer - 1
            with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
                if decoder_layer == 0:
                    # first decoder layer doesn't have skip connections
                    # since it is directly connected to the skip_layer
                    input = layers[-1]
                else:
                    input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

                rectified = tf.nn.relu(input)
                # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
                output = self.gen_deconv(rectified, out_channels)
                output = self.batchnorm(output)

                if dropout > 0.0:
                    output = tf.nn.dropout(output, keep_prob=1 - dropout)

                layers.append(output)

        # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
        with tf.variable_scope("decoder_1"):
            input = tf.concat([layers[-1], layers[0]], axis=3)
            rectified = tf.nn.relu(input)
            output = self.gen_deconv(rectified, generator_outputs_channels)
            output = tf.tanh(output)
            layers.append(output)

        return layers[-1]
    def create_model(self,):
        my_inputs = tf.placeholder(dtype=np.uint8,shape=(100,3202))

        my_img = my_inputs[:, :, tf.newaxis]
        my_img = tf.image.decode_jpeg(tf.image.encode_jpeg(my_img))
        my_img = tf.image.convert_image_dtype(my_img, dtype=tf.float32)
        my_img = tf.identity(my_img)
        my_img.set_shape([None, None, 1])

        inputs = self.preprocess(my_img[:, 1601:, :])
        #targets = self.preprocess(my_img[:, :1601, :])
        inputs = tf.image.resize_images(inputs, [256, 256], method=tf.image.ResizeMethod.AREA)
        #targets = tf.image.resize_images(targets, [256, 256], method=tf.image.ResizeMethod.AREA)
        #inputs, targets = tf.train.batch([inputs, targets], batch_size=1)
        inputs = inputs[tf.newaxis,:,:,:]
        #targets = targets[tf.newaxis,:,:,:]

        def create_discriminator(discrim_inputs, discrim_targets):
            n_layers = 3
            layers = []
            # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
            input = tf.concat([discrim_inputs, discrim_targets], axis=3)

            # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
            with tf.variable_scope("layer_1"):
                convolved = self.discrim_conv(input, 64, stride=2)
                rectified = self.lrelu(convolved, 0.2)
                layers.append(rectified)

            # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
            # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
            # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
            for i in range(n_layers):
                with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                    out_channels = 64 * min(2**(i+1), 8)
                    stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                    convolved = self.discrim_conv(layers[-1], out_channels, stride=stride)
                    normalized = self.batchnorm(convolved)
                    rectified = self.lrelu(normalized, 0.2)
                    layers.append(rectified)

            # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                convolved = self.discrim_conv(rectified, out_channels=1, stride=1)
                output = tf.sigmoid(convolved)
                layers.append(output)

            return layers[-1]

        with tf.variable_scope("generator"):
            out_channels = 1#int(targets.shape[-1])
            outputs = self.create_generator(inputs, out_channels)

        # create two copies of discriminator, one for real pairs and one for fake pairs
        # they share the same underlying variables
        """
        with tf.name_scope("real_discriminator"):
            with tf.variable_scope("discriminator"):
                # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
                predict_real = create_discriminator(inputs, targets)
        
        with tf.name_scope("fake_discriminator"):
            with tf.variable_scope("discriminator", reuse=True):
                # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
                predict_fake = create_discriminator(inputs, outputs)
        
        with tf.name_scope("discriminator_loss"):
            # minimizing -tf.log will try to get inputs to 1
            # predict_real => 1
            # predict_fake => 0
            discrim_loss = tf.reduce_mean(-(tf.log(predict_real + self.EPS) + tf.log(1 - predict_fake + self.EPS)))
        
        
        with tf.name_scope("generator_loss"):
            # predict_fake => 1
            # abs(targets - outputs) => 0
            gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + self.EPS))
            gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
            gen_loss = gen_loss_GAN * 1.0 + gen_loss_L1 * 100.0
        
        with tf.name_scope("discriminator_train"):
            discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
            discrim_optim = tf.train.AdamOptimizer(0.0002, 0.5)
            discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
            discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

        with tf.name_scope("generator_train"):
            with tf.control_dependencies([discrim_train]):
                gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
                gen_optim = tf.train.AdamOptimizer(0.0002, 0.5)
                gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
                gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

        ema = tf.train.ExponentialMovingAverage(decay=0.99)
        update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

        global_step = tf.train.get_or_create_global_step()
        incr_global_step = tf.assign(global_step, global_step+1)
        """
        return self.Model(
            my_inputs = my_inputs,
            #predict_real=predict_real,
            #predict_fake=predict_fake,
            #discrim_loss=ema.average(discrim_loss),
            #discrim_grads_and_vars=discrim_grads_and_vars,
            #gen_loss_GAN=ema.average(gen_loss_GAN),
            #gen_loss_L1=ema.average(gen_loss_L1),
            #gen_grads_and_vars=gen_grads_and_vars,
            outputs=outputs,
            #train=tf.group(update_losses, incr_global_step, gen_train),
        )
    def convert(self,image):
        size = [self.CROP_SIZE, int(round(self.CROP_SIZE * 1))]
        image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)
        return tf.image.convert_image_dtype(image, dtype=tf.float32, saturate=True)
    def create_sweep(self,):
        sweep = np.zeros(shape=(100,1601))
        self.row=0
        for i in range(self.lines):
            if i - self.last_interval == 0:
                #self.interval = np.random.randint(2,11)
                self.last_interval += self.interval
                self.row += 1
            randint = np.random.randint(0,40)
            sweep[i,self.action_space[self.row%self.n_actions]-10-randint:self.action_space[self.row%self.n_actions]+11+randint]=1

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

        l=np.array(sweep[0:100,:],)
        l = l * 255 / 3

        d=np.zeros(shape=(100,1601),)
        d=np.array(d[0:100,:])
        image = np.concatenate((d, l), axis=1)
        return image

    def generate_img(self,input,):
        tf.reset_default_graph()
        my_img = input

        model = self.create_model()

        outputs = self.deprocess(model.outputs)
        converted_outputs = self.convert(outputs)

        saver = tf.train.Saver()
        sv = tf.train.Supervisor(logdir=None, save_summaries_secs=0, saver=None)
        # saver = sv.saver
        checkpoint = tf.train.latest_checkpoint("facades_train")
        with sv.managed_session() as sess:
            print("loading model from checkpoint")
            saver.restore(sess, checkpoint)
            img = sess.run(converted_outputs,feed_dict={model.my_inputs:my_img})

            img = img * 255
            img = np.reshape(img, (256, 256))
            img = transform.resize(img, (100, 1601))
            #cv2.imwrite("haha.jpg",img)
            plt.imshow(img, interpolation='nearest', aspect='auto')
            plt.show()


        return img

""""""
if __name__ == "__main__":
    gen = generator()
    img = gen.create_sweep()
    gen_img = gen.generate_img(img)
    
    
    """
    saver = tf.train.Saver()
    sv = tf.train.Supervisor(logdir=None, save_summaries_secs=0, saver=None)
    checkpoint = tf.train.latest_checkpoint("facades_train")
    
    with sv.managed_session() as sess:
        print("loading model from checkpoint")
        saver.restore(sess, checkpoint)
        img = sess.run(gen_img)
    
    img = img * 255
    img = np.reshape(img, (256, 256))
    img = transform.resize(img, (100, 1601))
    plt.imshow(img, interpolation='nearest', aspect='auto')
    plt.show()
    """