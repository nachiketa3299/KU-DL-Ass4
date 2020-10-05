import os
import pickle


class Preset:

    def __init__(self, preset_num=1):

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

        def setTrainingParameters(_batch_size, _num_epochs, _evaluate_every, _checkpoint_every, _num_checkpoints,
                                  _data_augmentation):
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
            setModelParameters(0.1, 1, 0.001, 0.1, 5, 10)
            setTrainingParameters(128, 200, 100, 100, 3, True)
        # chane lr decay to 0.96 (1: 1.0)
        elif preset_num == 2:
            setModelParameters(0.1, 0.96, 0.001, 0.1, 5, 10)
            setTrainingParameters(128, 200, 100, 100, 3, True)
        # change num_resicual_units to 3 (2: 5)
        elif preset_num == 3:
            setModelParameters(0.1, 0.96, 0.001, 0.1, 3, 10)
            setTrainingParameters(128, 200, 100, 100, 3, True)
        # base: 2, change num_residual_units to 9
        elif preset_num == 4:
            setModelParameters(0.1, 0.96, 0.001, 0.1, 9, 10)
            setTrainingParameters(128, 200, 100, 100, 3, True)

        # base 4, change lr_decay_rate  to 0.9
        elif preset_num == 5:
            setModelParameters(0.1, 0.90, 0.001, 0.1, 9, 10)
            setTrainingParameters(128, 200, 100, 100, 3, True)

    def saveCurPresetToPickle(self, directory="./INFOS/pickles"):
        if not os.path.exists(directory):
            os.makedirs(directory)
        picklename = os.path.join(directory, f'{self.preset}_{self.timestamp}.pickle')
        with open(picklename, 'wb') as pckl:
            pickle.dump(self, pckl)
        print(f"> Preset {self.preset}번의 학습정보가 {picklename}에 저장되었습니다.")

    def saveCurTrainResultToTxt(self, directory="./INFOS", name='train_overall.txt'):
        if not os.path.exists(directory):
            os.makedirs(directory)
        fname = os.path.join(directory, name)
        if not os.path.isfile(fname):
            with open(fname, 'w') as _f:
                pass
        with open(fname, 'a') as ttxt:
            s = ''
            for key in self.__dict__.keys():
                s += str(self.__dict__[key]) + '\t'
            s += '\n'
            ttxt.write(s)
        print(f"> Preset {self.preset}번의 학습기록이 {fname}에 기록되었습니다.")

    @classmethod
    def readAllPickles(cls):
        pickle_dir = os.path.join(os.path.curdir, 'INFOS', 'pickles')
        pickle_list = os.listdir(pickle_dir)
        rl = []
        for pckl in pickle_list:
            if pckl.split('.')[-1] == 'pickle':
                pckl_file_dir = os.path.join(pickle_dir, pckl)
                with open(pckl_file_dir, 'rb') as pickle_object:
                    rl.append(pickle.load(pickle_object))
        print(rl)
        return rl

    @classmethod
    def readCertainPickle(cls, timestamp):
        pickle_dir = os.path.join(os.path.curdir, 'INFOS', 'pickles')
        pickle_list = os.listdir(pickle_dir)
        rl = []
        for pckl in pickle_list:
            if pckl.split('.')[-1] == 'pickle':
                pckl_file_dir = os.path.join(pickle_dir, pckl)
                with open(pckl_file_dir, 'rb') as pickle_object:
                    rl.append(pickle.load(pickle_object))
        print(rl)
        for pobj in rl:
            if str(pobj.timestamp) == timestamp:
                return pobj

    @classmethod
    def del_all_flags(cls, _FLAGS):
        flags_dict = _FLAGS._flags()
        keys_list = [keys for keys in flags_dict]
        for keys in keys_list:
            _FLAGS.__delattr__(keys)

