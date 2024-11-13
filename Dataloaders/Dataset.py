import os, re, pickle, torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import category_encoders as ce
from sklearn.preprocessing import RobustScaler, MinMaxScaler

import warnings
warnings.filterwarnings(action='ignore')

class SSDataset(Dataset):
    def __init__(self, configs, mode:str='train'):
        self.configs = configs
        self.mode = mode
        data = self.load_data()
        self.ET_X, self.ET_y, self.ET_id, self.ET_X_eqp = data

        if isinstance(self.ET_X, np.ndarray):
            self.ET_X = torch.from_numpy(self.ET_X).float()
            self.ET_y = torch.from_numpy(self.ET_y).float()
            self.ET_X_eqp = torch.from_numpy(self.ET_X_eqp).float()

    def __getitem__(self, idx):
        return self.ET_X[idx].float(), self.ET_X_eqp[idx].float(), self.ET_y[idx].float(), self.ET_id[idx]
    
    def __len__(self):
        return len(self.ET_y)
    
    @property
    def input_dim(self):
        return self.ET_X.shape[-1]
    
    @property
    def num_out(self):
        return self.ET_y.shape[-1]
    
    def load_data(self):
        data_dir = os.path.join(self.configs.data_dir,
                                f"{self.configs.encoding_type.upper()}", 
                                f"{self.configs.NP}".upper())
        et_data = self.preprocess(data_dir) if not os.path.exists(os.path.join(data_dir, "ET_data.pkl")) else pd.read_pickle(os.path.join(data_dir, "ET_data.pkl"))

        valid_len = int(et_data['x_train'].shape[0]*(self.configs.valid_ratio))
        if self.mode == "train":
            ET_X, ET_y, ET_id = et_data['x_train'][:-valid_len, :], et_data['y_train'][:-valid_len, :], et_data['train_wafer_id'][:-valid_len]
            ET_X_eqp = et_data['eqp_train'][:-valid_len, :]
        elif self.mode == "valid":
            ET_X, ET_y, ET_id = et_data['x_train'][-valid_len:, :], et_data['y_train'][-valid_len:, :], et_data['train_wafer_id'][-valid_len:]
            ET_X_eqp = et_data['eqp_train'][-valid_len:, :]
        elif self.mode == "test":
            ET_X, ET_y, ET_id = et_data['x_test'], et_data['y_test'], et_data['test_wafer_id']
            ET_X_eqp = et_data['eqp_test']
        else:
            raise ValueError("'mode' should be one of ['train', 'valid', 'test']")
        
        return ET_X, ET_y, ET_id, ET_X_eqp

    def preprocess(self, data_dir):
        if not os.path.exists(data_dir): os.makedirs(data_dir, exist_ok=True)
        
        # load original data
        _df = pd.read_csv(os.path.join(self.configs.data_dir, 'df_0725.csv'))
        df = _df.iloc[:, 2:]

        wafer_id = np.array([str(i)+'_'+str(j) for (i, j) in _df.iloc[:, :2].values])

        # define cols
        ET_col = [col for col in df.columns if 'y_' in col]
        if self.configs.NP == "N":
            ET_col = [col for col in ET_col if 'P' not in col]  # 30
        elif self.configs.NP == "P":
            ET_col = [col for col in ET_col if 'P' in col]      # 18

        """Split train/test set"""
        train_idx = np.arange(df.shape[0])[:-int(df.shape[0]*(self.configs.test_ratio))]  # 333
        # train_idx, valid_idx = _train_idx[:-int(len(_train_idx)*(self.configs.test_ratio))], _train_idx[-int(len(_train_idx)*(self.configs.test_ratio)):]
        test_idx  = np.arange(df.shape[0])[-int(df.shape[0]*(self.configs.test_ratio)):]  # 83
        valid_idx = test_idx

        """Make ET data"""
        # define x cols to predict ET
        im_num_step_list = list(set([col.split('_')[0]+'_'+col.split('_')[1]+'_' for col in df.columns if any(excl in col for excl in ['IM', '_num'])]))  # 89
        et_x_col = [col for col in df.columns if any(step in col for step in im_num_step_list)]
        eqp_col = [col for col in et_x_col if 'eqp' in col]
        im_num_col = [col for col in et_x_col if 'eqp' not in col]

        train, valid, test= [], [], []
        for col in [im_num_col, eqp_col]:
            x_train, y_train = df[col].iloc[train_idx], df[ET_col].iloc[train_idx]
            x_valid, y_valid = df[col].iloc[valid_idx], df[ET_col].iloc[valid_idx]
            x_test, y_test = df[col].iloc[test_idx], df[ET_col].iloc[test_idx]

            # encoding
            x_train_enc, x_valid_enc, x_test_enc = self.encoding(x_train, y_train, x_valid, x_test)

            # scaling
            x_train_scaled, x_valid_scaled, x_test_scaled = self.scaling(x_train_enc, x_valid_enc, x_test_enc)
            x_train_scaled = pd.DataFrame(x_train_scaled, columns = list(self.scaler.get_feature_names_out()))
            x_valid_scaled = pd.DataFrame(x_valid_scaled, columns = list(self.scaler.get_feature_names_out()))
            x_test_scaled = pd.DataFrame(x_test_scaled, columns = list(self.scaler.get_feature_names_out()))

            train.append(x_train_scaled)
            valid.append(x_valid_scaled)
            test.append(x_test_scaled)

        et_data = self.make_dataset(train[0], y_train,
                                     valid[0], y_valid,
                                     test[0], y_test,
                                     wafer_id)
        et_dict = {
                   'eqp_train':train[1].values,
                   'eqp_valid':valid[1].values,
                   'eqp_test':test[1].values}
        et_data.update(et_dict)

        """save dataset"""       
        with open(os.path.join(data_dir, 'ET_data.pkl'), "wb") as f:
            pickle.dump(et_data, f, pickle.HIGHEST_PROTOCOL)
        
        return et_data

    def scaling(self, x_train, x_valid, x_test):
        # self.scaler = RobustScaler()
        self.scaler = MinMaxScaler(feature_range=(1e-6, 1))
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_valid_scaled = self.scaler.transform(x_valid)
        x_test_scaled = self.scaler.transform(x_test)
        return x_train_scaled, x_valid_scaled, x_test_scaled

    def encoding(self, x_train: pd.DataFrame, y_train: pd.DataFrame, 
                 x_valid: pd.DataFrame, x_test: pd.DataFrame):
        encoder_class = getattr(ce, f"{self.configs.encoding_type}Encoder", None)
        if encoder_class:
            encoder = encoder_class(return_df=True)
            print(f"Using {self.configs.encoding_type} encoding")
        else:
            NotImplementedError(f"Unsupported encoding type: {self.configs.encoding_type}")

        encode_cols = [col for col in x_train.columns if 'dose' not in col and 'VM' not in col]
        numeric_cols = [col for col in x_train.columns if 'dose' in col or 'VM' in col]

        x_train_enc = encoder.fit_transform(x_train[encode_cols].astype(str),
                                            np.nan_to_num(y_train.values).mean(axis=1),)
        x_valid_enc = encoder.transform(x_valid[encode_cols].astype(str))
        x_test_enc = encoder.transform(x_test[encode_cols].astype(str))

        # CONCAT [encoding columns, numeric columns]
        x_train_ = pd.concat([x_train[numeric_cols], x_train_enc], axis=1)
        x_valid_, x_test_ = pd.concat([x_valid[numeric_cols], x_valid_enc], axis=1), pd.concat([x_test[numeric_cols], x_test_enc], axis=1)
        
        # columns step 순 재정렬
        if self.configs.encoding_type != "OneHot":
            x_train_enc, x_valid_enc, x_test_enc = x_train_[x_train.columns], x_valid_[x_valid.columns], x_test_[x_test.columns]
        else:
            sorted_columns = sorted(x_train_.columns, key=lambda x: int(x.split('_')[1]))
            x_train_enc, x_valid_enc, x_test_enc = x_train_[sorted_columns], x_valid_[sorted_columns], x_test_[sorted_columns]

        return x_train_enc, x_valid_enc, x_test_enc

    @staticmethod
    def make_dataset(x_train, y_train, x_valid, y_valid, x_test, y_test, wafer_id):
        data = dict()
        data['x_train'] = x_train.values
        data['y_train'] = y_train.values
        data['x_valid'] = x_valid.values
        data['y_valid'] = y_valid.values
        data['x_test']  = x_test.values
        data['y_test']  = y_test.values

        data['train_wafer_id'] = wafer_id[x_train.index]
        data['valid_wafer_id'] = wafer_id[x_valid.index]
        data['test_wafer_id']  = wafer_id[x_test.index]
        return data
    

if __name__ == '__main__':
    import sys, argparse
    from os import path
    sys.path.append(path.dirname( path.dirname( path.abspath(__file__) ) ))
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./Data/')
    parser.add_argument('--VM', type=str, default='all', choices=('all', 'last'))
    parser.add_argument('--pad-type', type=str, default="zero", choices=('zero', 'replicate'))
    parser.add_argument('--encoding-type', type=str, default='Count', choices=('Count', 'Target', 'CatBoost', 'OneHot'))
    parser.add_argument('--backbone', type=str, default='CNN')
    parser.add_argument('--test-ratio', default=.2, type=float)
    parser.add_argument('--valid-ratio', default=.2, type=float)
    args = parser.parse_args()

    dataset = SSDataset(args)
    dataset.load_data()