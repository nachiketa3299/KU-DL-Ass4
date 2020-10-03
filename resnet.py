import tensorflow as tf
import numpy as np
from tensorflow.python.training import moving_averages

##############################################################################################################
#                    TODO : X1 ~ X10에 올바른 숫자 또는 변수를 채워넣어 ResNet32 코드를 완성할 것                 #
##############################################################################################################


class ResNet:
    def __init__(self, config):
        self._num_residual_units = config.num_residual_units #unit 개수, 즉 n의 크기
        self._batch_size = config.batch_size
        self._relu_leakiness = config.relu_leakiness
        self._num_classes = config.num_classes # label 개수 (10개-airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
        self._l2_reg_lambda = config.l2_reg_lambda

        self.X = tf.placeholder(tf.float32, [None, 32, 32, 3], name="X")
        self.Y = tf.placeholder(tf.float32, [None, self._num_classes], name="Y")
        self.extra_train_ops = [] # batch norm 파라미터 학습을 위한 operation 저장할 변수

        # TODO 1. 층별 filter 의 output channel 갯수 16(input) -> 16(X1) -> 32(X2) -> 64
        filters = [16,
                   16,   # X1
                   32,   # X2
                   64]

        self.FILTER_SIZE = 3 # filter size가 3이 아닌 경우가 없어서 그냥 선언함
        activate_before_residual = [True, False, False] # activation 연산 수행 시점의 residual 연산 전후 여부

        with tf.variable_scope('init'): # 최초 convolutional layer (input이 들어감)
            # TODO 2. Initial Convolution
            x = self._conv(name='init_conv',
                           x=self.X,
                           filter_size=self.FILTER_SIZE,                # X3: Conv시 필터 사이즈 (3)
                           in_filters=self.X.get_shape().as_list()[-1], # X4: 인풋 이미지의 채널 수 (3)
                           out_filters=filters[0],                      # X5: 다음 레이어의 채널 수 (16)
                           strides=[1, 1, 1, 1])                        # X6: 스트라이드는 한 칸 씩

        with tf.variable_scope('unit_1_0'):
            # 첫 번째 residual unit
            x = self._residual(x_in=x,
                               in_filter=filters[0],  # 16
                               out_filter=filters[1], # 16
                               activate_before_residual=activate_before_residual[0], # True
                               strides=[1, 1, 1, 1])

        for i in range(1, self._num_residual_units): #나머지 residual unit (n이 3이라면 두번째, 세번째 unit들)
            with tf.variable_scope('unit_1_%d' % i):
                x = self._residual(x_in=x,
                                   in_filter=filters[1], # 16
                                   out_filter=filters[1], # 16
                                   activate_before_residual=False,
                                   strides=[1, 1, 1, 1])

        with tf.variable_scope('unit_2_0'):
            # 여기서 채널 수가 증가함 (16 -> 32)
            # 내 생각인데 수축도 여기서 일어나야 함 (32, 32, 16) -> (16, 16, 32)
            x = self._residual(x_in=x,
                               in_filter=filters[1], # 16
                               out_filter=filters[2], # 32
                               activate_before_residual=activate_before_residual[1], # True
                               strides=[1, 2, 2, 1])
        for i in range(1, self._num_residual_units):
            with tf.variable_scope('unit_2_%d' % i):
                x = self._residual(x_in=x,
                                   in_filter=filters[2], # 32
                                   out_filter=filters[2], # 32
                                   activate_before_residual=False,
                                   strides=[1, 1, 1, 1])

        with tf.variable_scope('unit_3_0'):
            x = self._residual(x_in=x,
                               in_filter=filters[2], # 32
                               out_filter=filters[3], # 64
                               activate_before_residual=activate_before_residual[2], # True
                               strides=[1, 2, 2, 1])
        for i in range(1, self._num_residual_units):
            with tf.variable_scope('unit_3_%d' % i):
                x = self._residual(x_in=x,
                                   in_filter=filters[3], # 64
                                   out_filter=filters[3], # 64
                                   activate_before_residual=False,
                                   strides=[1, 1, 1, 1])

        with tf.variable_scope('unit_last'):
            x = self._batch_norm('final_bn', x)
            x = self._relu(x, self._relu_leakiness)
            x = self._global_avg_pool(x) # (?, 8, 8, 64) 크기의 feature maps을 average pooling을 통해 (?, 8, 8)로 요약 및 축소

        with tf.variable_scope('logit'):
            logits = self._fully_connected(x, self._num_classes) # fully connected layer
            self.predictions = tf.nn.softmax(logits)
            self.predictions = tf.argmax(self.predictions, 1, name="predictions")

        with tf.variable_scope('loss'):
            xent = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.Y)
            self.loss = tf.reduce_mean(xent, name='xent')
            self.loss += self._decay()

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters # he 초기화를 위한 크기 계산
            # filter size 가로, 세로, input channel, output channel 크기로 filter 초기화
            kernel = tf.get_variable(
                'DW', [filter_size, filter_size, in_filters, out_filters],
                tf.float32, initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
            return tf.nn.conv2d(x, kernel, strides, padding='SAME') # convolution 연산

    def _residual(self, x_in, in_filter, out_filter, activate_before_residual=False, strides=[1, 1, 1, 1]):
        if activate_before_residual:
            with tf.variable_scope('common_activation'):
                x = self._batch_norm('init_bn', x_in)
                x = self._relu(x, self._relu_leakiness)
                orig_x = x # skip connection 때 넘 겨줄 feature orig_x에 복사
        else:
            with tf.variable_scope('residual_activation'):
                orig_x = x_in # skip connection 때 넘 겨줄 feature orig_x에 복사
                x = self._batch_norm('init_bn', x_in)
                x = self._relu(x, self._relu_leakiness)

        # sub1
        # Conv2d, BatchNorm, ReLU (BatchNorm & ReLU are already calculated)
        with tf.variable_scope('sub1'):
            # TODO 2. SUB1 블록
            x = self._conv(name='conv1',
                           x=x,
                           filter_size=self.FILTER_SIZE,  # X7
                           in_filters=in_filter,
                           out_filters=out_filter,
                           strides=[1, 1, 1, 1])          # X8

        # sub2
        # Conv2d, BatchNorm, Shortcut Connection, ReLU
        with tf.variable_scope('sub2'):
            x = self._batch_norm('bn2', x)
            x = self._relu(x, self._relu_leakiness)
            # TODO 3. SUB2 블록
            x = self._conv(name='conv2',
                           x=x,
                           filter_size=self.FILTER_SIZE, # X9
                           in_filters=out_filter,
                           out_filters=out_filter,
                           strides=strides)         # X10

        with tf.variable_scope('sub_add'):
            if in_filter != out_filter: # stride 크기가 2일 때 channel 크기가 안맞는 경우 크기 조정을 통해 skip connection이 원활하게 조정
                orig_x = tf.nn.avg_pool(orig_x, strides, strides, 'VALID') # pooling으로 feature map 크기 맞추고
                orig_x = tf.pad(orig_x, [[0, 0], [0, 0], [0, 0], [(out_filter - in_filter) // 2, (out_filter - in_filter) // 2]]) # padding으로 채널 맞춤
            x += orig_x # skip connection

        tf.logging.debug('image after unit %s', x.get_shape())
        return x

    def _relu(self, x, leakiness=0.0):
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def _batch_norm(self, name, x):
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]

            beta = tf.get_variable(
                'beta', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable(
                'gamma', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32))

            # tf.nn.moments calculate the mean and variance of x
            # tf.nn.moments(x, axes, shift=None, keepdims=False, name=None)
            # axes => array of ints, axes along which to compute mean and variance
            mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

            moving_mean = tf.get_variable(
                'moving_mean', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32),
                trainable=False)
            moving_variance = tf.get_variable(
                'moving_variance', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32),
                trainable=False)

            self.extra_train_ops.append(moving_averages.assign_moving_average(
                moving_mean, mean, 0.9))
            self.extra_train_ops.append(moving_averages.assign_moving_average(
                moving_variance, variance, 0.9))

            y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
            y.set_shape(x.get_shape())
            return y

    def _fully_connected(self, x, out_dim):
        dim = tf.reduce_prod(x.get_shape()[1:]).eval()
        print(f"- dim: {dim}")
        x = tf.reshape(x, [-1, dim])
        w = tf.get_variable(
            'DW', [dim, out_dim],
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        b = tf.get_variable('biases', [out_dim],
                            initializer=tf.constant_initializer())
        return tf.nn.xw_plus_b(x, w, b)

    def _global_avg_pool(self, x):
        # (?, 8, 8, 64) 크기의 feature maps을 average pooling을 통해 (?, 1, 1, 64)로 요약 및 축소
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])

    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables(): # 학습 가능한 모든 변수들의
            if var.op.name.find(r'DW') > 0:
                costs.append(tf.nn.l2_loss(var)) # l2 norm을 구해서 costs에 저장

        return tf.multiply(self._l2_reg_lambda, tf.add_n(costs)) # costs안에 값들을 모두 더한 뒤 lambda 값에 곱해줌