class Preset:

    def __init__(self, preset_num=1):
        # not used
        self.SEED = 4

        # preset number (initialized in train)
        self.preset = preset_num

        # not changed
        self.model = 'ResNet'
        self.optimizer = 'SGD (Momentum)'
        self.activation_function = 'ReLU'
        self.weight_initialization = 'He_normarl'
        self.normalization = 'Batch Norm'

        ## initialized with flags
        # model params
        self.lr = None
        self.lr_decay = None
        self.l2_reg_lambda = None
        self.relu_leakiness = None
        self.num_residual_units = None
        self.num_classes = None
        # training params
        self.batch_size = None
        self.num_epochs = None
        self.evaluate_every = None
        self.checkpoint_every = None
        self.num_checkpoints = None
        self.data_augmentation = None
        # misc params
        self.allow_soft_placement = None
        self.log_device_placement = None

        # results
        self.timestamp = None
        self.training_time = None
        self.early_stopping_epoch = None
        self.val_accuracy = None
        self.test_accuracy = None
        self.finished = False

        def setModelParameters(_lr, _lr_decay, _l2_reg_lambda, _relu_leakiness, _num_residual_units, _num_classes):
            self.lr = _lr
            self.lr_decay = _lr_decay
            self.l2_reg_lambda = _l2_reg_lambda
            self.relu_leakiness = _relu_leakiness
            self.num_residual_units = _num_residual_units
            self.num_classes = _num_classes

            self.model += str(6 * self.num_residual_units + 2)

        def setTrainingParameters(_batch_size, _num_epochs, _evaluate_every, _checkpoint_every, _num_checkpoints, _data_augmentation):
            self.batch_size = _batch_size
            self.num_epochs = _num_epochs
            self.evaluate_every = _evaluate_every
            self.checkpoint_every = _checkpoint_every
            self.num_checkpoints = _num_checkpoints
            self.data_augmentation = _data_augmentation

        def setMiscParameters(_allow_soft_placement, _log_device_placemnet):
            self.allow_soft_placement = _allow_soft_placement
            self.log_device_placement = _log_device_placemnet


        setMiscParameters(True, False)

        # if preset_num == 0: # Default Setting
        #     setModelParameters(0.1, 0.1, 0.0, 0.1, 5, 10)
        #     setTrainingParameters(64, 200, 100, 100, 5, True)

        if preset_num == 1:
            setModelParameters(0.1, 0.1, 0.001, 0.1, 3, 10)
            setTrainingParameters(128, 200, 100, 100, 3, True)

    @classmethod
    def del_all_flags(cls, _FLAGS):
        flags_dict = _FLAGS._flags()
        keys_list = [keys for keys in flags_dict]
        for keys in keys_list:
            _FLAGS.__delattr__(keys)
