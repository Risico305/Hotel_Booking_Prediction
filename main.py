import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from imblearn.over_sampling import SMOTE
from keras import Input, Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import LSTM, Dense
from keras.models import load_model
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from numpy.random import seed


seed(1337)


class Booking:
    def __init__(self, is_LSTM=False, nb_epoch=500, lr=0.001, batch_size=100, patience=20):
        self.is_LSTM = is_LSTM
        self.nb_epoch = nb_epoch
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience

    def __read_data(self, path, is_Test=False):
        dataset = pd.read_csv(path)
        ids = dataset['ID'].copy()
        dataset = dataset.drop(labels=['ID'
            , 'entry_page'
            , 'device_type_name'
            , 'prior_session'
            , 'browser'
            , 'search_type'], axis=1)
        if is_Test == True:
            return dataset, ids
        else:
            lstm_seq = dataset
            dataset['booker'] = dataset['booker'] * 1
            labels = dataset['booker']
            return dataset, labels, lstm_seq

    def __build_correlaions(self, dataset):
        correlation = dataset.corr()
        plt.figure(figsize=(5, 5))
        sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='viridis')
        plt.xticks(range(len(correlation.columns)), correlation.columns)
        plt.yticks(range(len(correlation.columns)), correlation.columns)
        plt.show()

    def __varian_plot(self, x):

        pca = PCA(n_components=13)
        pca.fit(x)
        var = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3) * 100)
        plt.ylabel('% Variance Explained')
        plt.xlabel('# of Features')
        plt.title('PCA Analysis')
        plt.style.context('seaborn-whitegrid')
        plt.plot(var)
        plt.show()

    def __over_sample(self, X, Y=None):
        smt = SMOTE(sampling_strategy='auto')
        _X, _Y = smt.fit_sample(X, Y)
        return _X, _Y

    def __scale_input(self, X, is_LSTM=False):

        if is_LSTM == True:
            scaler = MinMaxScaler()
            norm_X = scaler.fit_transform(X)
            return norm_X
        else:
            scaler = StandardScaler()
            norm_X = scaler.fit_transform(X)
            return norm_X

    def __build_sequence(self, dataset, n_in=1, n_out=1, dropnan=True):
        num_vars = 1 if type(dataset) is list else dataset.shape[1]
        dataframe = pd.DataFrame(dataset)
        cols, names = list(), list()
        for i in range(n_in, 0, -1):
            cols.append(dataframe.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(num_vars)]
        for i in range(0, n_out):
            cols.append(dataframe.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(num_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(num_vars)]
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # NaN?
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    def __matthews_correlation(self, y_true, y_pred):
        y_pred_pos = K.round(K.clip(y_pred, 0, 1))
        y_pred_neg = 1 - y_pred_pos

        y_pos = K.round(K.clip(y_true, 0, 1))
        y_neg = 1 - y_pos

        tp = K.sum(y_pos * y_pred_pos)
        tn = K.sum(y_neg * y_pred_neg)

        fp = K.sum(y_neg * y_pred_pos)
        fn = K.sum(y_pos * y_pred_neg)

        numerator = (tp * tn - fp * fn)
        denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        return numerator / (denominator + K.epsilon())

    def __neural(self, X_train, y_train, x_val, y_val):

        # X_train, X_test, y_train, y_test = train_test_split(scaled, Y_sm, test_size=0.3,
        #  random_state=1337)
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        x_val = x_val.reshape((x_val.shape[0], 1, x_val.shape[1]))

        input = Input(shape=(X_train.shape[1], X_train.shape[2]))
        x = LSTM(units=100, recurrent_dropout=0.2, dropout=0.2)(input)
        out = Dense(1, activation="sigmoid")(x)

        model = Model(input, out)
        checkpointer = ModelCheckpoint(filepath='models/model.hdf5',
                                       verbose=1,
                                       save_best_only=True)
        earlystopper = EarlyStopping(monitor='val_loss',
                                     patience=self.patience,
                                     verbose=1)
        # rms = RMSprop(lr=0.001, rho=0.9,epsilon=None, decay=0.0)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[self.__matthews_correlation])
        model.summary()
        history = model.fit(X_train, y_train, callbacks=[checkpointer, earlystopper], validation_data=(x_val, y_val),
                            epochs=self.nb_epoch, batch_size=self.batch_size)
        return model, history

    def __classification(self, X_train, new_y, X_test, ids):

        rf = RandomForestClassifier(max_depth=2)
        tree = ExtraTreesClassifier()
        clf = sklearn.svm.NuSVC(gamma='auto')
        models = {'Random_Forest': rf, 'svm': clf, 'ETC': tree}
        for name, model in models.items():
            print('model %s training' % name)
            pca = PCA()
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
            model.fit(X_train, new_y)
            y_pred = model.predict(X_test)
            # print(sklearn.metrics.matthews_corrcoef(y_test, y_pred, sample_weight=None))
            dObj = pd.DataFrame(ids)
            dObj['label'] = list(y_pred)
            dObj.to_csv('pred_%s.csv', sep='\t', encoding='utf-8') % name

    def run(self):
        train_data, labels, train_seq = self.__read_data('data/train.csv')
        test_data, ids = self.__read_data('data/test.csv', is_Test=True)
        Y = labels
        X = train_data.drop(labels=['booker'], axis=1)
        new_X, new_y = self.__over_sample(X, Y)
        X_train = self.__scale_input(new_X, is_LSTM=False)
        X_test = self.__scale_input(test_data)
        self.__classification(X_train, new_y, X_test, ids)

        if self.is_LSTM == True:
            test_sequence = self.__build_sequence(test_data)
            X_test = self.__scale_input(test_sequence, is_LSTM=True)
            model = load_model('models/model.hdf5',
                               custom_objects={'__matthews_correlation': self.__matthews_correlation})
            model.summary()

            X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
            predict = model.predict(X_test, batch_size=self.batch_size)
            dObj = pd.DataFrame(ids)
            dObj['label'] = list(predict)
            dObj.to_csv('pred_lstm.csv', sep='\t', encoding='utf-8')


if __name__ == '__main__':
    book = Booking(is_LSTM=False)
    book.run()
