# load dependacies
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from lib.globalvar import ENTIRE_NUMBER, TOT_NUMBER_OF_GTH 


class DataLoader:
    """
    Arguments:
    - data_dir : Data directory which contains files mentioned above.
    - split_ratio : to decide length of training / test
    """

    def __init__(self, data_dir, training_length, window_prev, mode, from_pos=0, to_pos=0, verbose=False):
        self.window_prev = window_prev
        self.data_dir = data_dir
        self.split_ratio = training_length
        self.mode = mode
        self._from_pos = from_pos
        self._to_pos = to_pos        
        self._verbose = verbose
        train_X, test_X, train_y, test_y = self.preproc_entire()
        self.train_X = train_X
        self.test_X = test_X
        self.train_Y = train_y
        self.test_Y = test_y

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, val):
        self._verbose = val

    @property
    def from_pos(self) -> int:
        return self._from_pos

    @from_pos.setter
    def from_pos(self, val: int):
        self._from_pos = val

    @property
    def to_pos(self) -> int:
        return self._to_pos

    @to_pos.setter
    def to_pos(self, val: int):
        self._to_pos = val

    def preproc_entire(self):
        """ preproc_entire """
        dataset = self.preproc_csv()
        reframed = self.preproc_data_for_supervised(dataset,
                                                    self.window_prev,
                                                    1,
                                                    dropnan=True,
                                                    fillnan = False)
        values = reframed.values
        # train_length = int(self.split_ratio * len(dataset))
        train_length = len(dataset)-self.window_prev-1
        train = values[:train_length, :]
        if self.mode == 'predict':
            test = values[:len(dataset)-self.window_prev-1, :]
        else:
            if self.split_ratio == 1:
                train_length = (self.split_ratio - 1) * len(dataset)
                test = values[train_length:train_length + 1, :]
            else:
                test = values[train_length:train_length + 1, :]
        # split into input and outputs
        n_obs =  self.window_prev * ENTIRE_NUMBER
        train_X, train_y = train[:, :n_obs], train[:, -ENTIRE_NUMBER:]
        test_X, test_y = test[:, :n_obs], test[:, -ENTIRE_NUMBER:]
        # reshape input to be 3D [samples, timesteps, features]
        train_X = train_X.reshape((train_X.shape[0], self.window_prev, ENTIRE_NUMBER ))
        test_X = test_X.reshape((test_X.shape[0], self.window_prev, ENTIRE_NUMBER))
        return train_X, test_X, train_y, test_y

    def preproc_csv(self):
        """ preproc_csv """
        # read .csv raw file from datadirectory
        rawdata = pd.read_csv(self.data_dir)        
        rawdata = rawdata.sort_values(by=['round'], axis=0) # sort with ascending order

        #from sklearn.preprocessing import MinMaxScaler
        #scaler = MinMaxScaler(feature_range=(0,1))
        #rawdata = scaler.fit_transform(rawdata)
        #rawdata = pd.DataFrame(rawdata, columns=['round', '1', '2', '3', '4', '5', '6'])
        
        raw_np = rawdata.to_numpy()
        if self._verbose:
            print(f'DataLoader.preproc_csv origin = {len(raw_np)}, {raw_np}')
        raw_np = raw_np[self._from_pos:,1:]
        raw_np_proc = raw_np[:,1:]
        if self._verbose:
            print(f'rDataLoader.preproc_csv aw_np_proc = {len(raw_np_proc)}, {raw_np_proc}')
        
        # to construct one-hot encoded input dataset
        inputnp = np.zeros((len(raw_np), ENTIRE_NUMBER))
        i = 0
        for row in raw_np_proc:
            for elem in row:
                #assign one-hot values
                inputnp[i, elem-1] = 1
            i += 1
        
        return inputnp
    
    def preproc_data_for_supervised(self, data, n_in, n_out=1, dropnan = False, fillnan = True):
        '''        

        Parameters
        ----------
        data : TYPE
            numpy array type time series data
        n_in : TYPE, optional
            DESCRIPTION. The default is 1.
        n_out : TYPE, optional
            DESCRIPTION. The default is 1.
        dropnan : TYPE, optional
            shifting values will generate nan value. handling the nan.

        '''
        n_in = self.window_prev
        
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
                
        # put it all together
        aggregated = pd.concat(cols, axis=1)
        aggregated.columns = names
        # drop rows with NaN values
        if dropnan:
            aggregated.dropna(inplace=True)
        if fillnan:
            aggregated.fillna(method = 'ffill')
        
        return aggregated
