class Preset:

    def __init__(self, preset_num=1):
        self.SEED = 4
        self.preset = preset_num
        self.model = None
        self.optimizer = None
        self.activation_function = None
        self.weight_initialization = None
        self.normalization = None
        self.lr = None
        self.lr_decay = None
        self.l2_reg_lambda = None
        self.relu_leakiness = None
        self.num_residual_units = None
        self.num_classes = None
        self.batch_size = None
        self.num_epochs = None
        self.evaluate_every = None
        self.checkpoint_every = None
        self.num_checkpoints = None
        self.data_augmentation = None
        self.allow_soft_placement = None
        self.log_device_placement = None

        def setCustomParameters(_model, _optimizer, _activation_function, _weight_initialization, _normalization):
            self.model = _model
            self.optimizer = _optimizer
            self.activation_function = _activation_function
            self.weight_initialization = _weight_initialization
            self.normalization = _normalization

        def setModelParameters(_lr, _lr_decay, _l2_reg_lambda, _relu_leakiness, _num_residual_units, _num_classes):
            self.lr = _lr
            self.lr_decay = _lr_decay
            self.l2_reg_lambda = _l2_reg_lambda
            self.relu_leakiness = _relu_leakiness
            self.num_residual_units = _num_residual_units
            self.num_classes = _num_classes

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

        if preset_num == 0: # Default Setting
            setCustomParameters(32, 'sgd', 'relu', 'he', 'batch')
            setModelParameters(0.1, 0.1, 0.0, 0.1, 5, 10)
            setTrainingParameters(64, 200, 100, 100, 5, True)

        elif preset_num == 1:
            setCustomParameters(32, 'sgd', 'relu', 'he', 'batch')
            setModelParameters(0.1, 0.1, 0.001, 0.1, 3, 10)
            setTrainingParameters(128, 200, 100, 100, 3, True)
