class Preset:

    def __init__(self, preset_num=1):
        self.SEED = 1888
        self.batch_size = None
        self.activation_function = None
        self.weight_initialization = None
        self.optimizer = None
        self.starter_learning_rate = None
        self.num_epochs = None
        self.keep_prob = None
        self.l2_reg_lambda = None
        self.data_augmentation = None
        self.learning_rate_decay_rate = None
        self.learning_rate_decay_step = None
        def setParameters(_batch_size, _activation_function, _weight_initialization, _optimizer, _learning_rate,
                          _num_epochs, _dropout, _weight_decay, _data_augmentation, _lr_decay_r, _lr_decay_s):
            self.batch_size = _batch_size
            self.activation_function = _activation_function
            self.weight_initialization = _weight_initialization
            self.optimizer = _optimizer
            self.starter_learning_rate = _learning_rate
            self.num_epochs = _num_epochs
            self.keep_prob = _dropout
            self.l2_reg_lambda = _weight_decay
            self.data_augmentation = _data_augmentation
            self.learning_rate_decay_rate = _lr_decay_r
            self.learning_rate_decay_step = _lr_decay_s

        if preset_num == 1:
            setParameters(128, 'relu', 0.01, 'adam', 0.001, 100, 0.9, 0.0, False, 1.0, 0)
