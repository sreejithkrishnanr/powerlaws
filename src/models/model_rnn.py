import numpy as np
from tqdm import tqdm

from keras.layers import Dense, Dropout, TimeDistributed, GRU
from keras.models import Sequential


def _generate_train_forecast_ts(x, y, output_window_size, pad=True, stride=1):
    assert x.shape[0] == y.shape[0]

    min_size = output_window_size
    input_size = x.shape[0]

    if input_size < min_size and pad:
        num_pads = min_size - input_size

        x_pad = np.zeros((num_pads, x.shape[1]))
        y_pad = np.zeros((num_pads, y.shape[1]))
        x = np.vstack((x_pad, x))
        y = np.vstack((y_pad, y))

    assert x.shape[0] >= min_size

    num_samples = x.shape[0] - output_window_size + 1

    x_res = []
    y_res = []

    for i in range(0, num_samples, stride):
        x_res.append(x[i:i + output_window_size])
        y_res.append(y[i:i + output_window_size].reshape(-1, 1))

    return np.array(x_res), np.array(y_res)


def _generate_train_ts(x, y, forecast_ids, output_window_size, stride=1):
    assert x.shape[0] == y.shape[0]
    assert y.shape[0] == forecast_ids.shape[0]

    ids = np.unique(forecast_ids)

    agg_x = []
    agg_y = []
    for fid in tqdm(ids):
        fx = x[forecast_ids == fid, :]
        fy = y[forecast_ids == fid, :]

        fx_res, fy_res = _generate_train_forecast_ts(fx, fy, output_window_size, stride=stride)

        agg_x.append(fx_res)
        agg_y.append(fy_res)

    agg_x = np.concatenate(agg_x)
    agg_y = np.concatenate(agg_y)

    return agg_x, agg_y


def _model_build_lstm(x_shape, y_shape):
    model = Sequential()
    model.add(GRU(units=125, return_sequences=True, batch_input_shape=(None, x_shape[1], x_shape[2])))
    model.add(Dropout(0.2))
    model.add(GRU(units=125, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(units=1, activation='linear')))
    model.compile(loss='mse', optimizer='adam')

    return model


def rnn_evaluate_model():
    pass


def rnn_build_model():
    pass


def rnn_predict_model():
    pass


# def model_train(ds, output_filepath, output_window_size, frequency):
#     data = build_window_features(ds, output_window_size, frequency)
#
#     scaler_x = StandardScaler()
#     scaler_y = StandardScaler()
#
#     features = select_features(data, frequency)
#     x = features.drop(columns=['Consumption', 'ForecastId', 'obs_id'])
#     y = features['Consumption'].reshape(-1, 1)
#     groups = features['ForecastId']
#     ids = features['obs_id']
#
#     tx_train, ty_train, tg_train, tid_train, tx_test, ty_test, tg_test, tid_test = \
#         generate_test_train_split(x, y, groups, ids, output_window_size)
#
#     x_train, y_train = generate_train_ts(
#         scaler_x.fit_transform(tx_train), scaler_y.fit_transform(ty_train), tg_train, output_window_size)
#     x_test, y_test = generate_train_ts(
#         scaler_x.transform(tx_test), scaler_y.transform(ty_test), tg_test, output_window_size)
#
#     model = model_build_lstm(x_train.shape, y_train.shape)
#
#     history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=400, callbacks=[
#         ReduceLROnPlateau(patience=5),
#         EarlyStopping(patience=10),
#     ])
#
#     y_test_pred = model.predict(x_test)
#     x_train, _ = generate_train_ts(scaler_x.fit_transform(
#         tx_train), scaler_y.fit_transform(ty_train), tg_train, output_window_size, stride=output_window_size)
#     y_train_pred = model.predict(x_train)
#
#     epochs = history.epoch[-1]
#     model = model_build_lstm(x.shape, y.shape)
#
#     scaler_x = StandardScaler()
#     scaler_y = StandardScaler()
#     x, y = generate_train_ts(scaler_x.fit_transform(x), scaler_y.fit_transform(y), groups, output_window_size)
#     model.fit(x, y, epochs=epochs)
#
#     model.save(output_filepath)
#
#     return scaler_y.inverse_transform(y_test_pred.ravel()), tid_test, \
#            scaler_y.inverse_transform(y_train_pred.ravel()), tid_train

# dataset = pd.read_csv('data/processed/train_86400000000000.csv', parse_dates=[2])
# ds = dataset.loc[dataset['SiteId'] == 64, :]
# y_test_pred, id_test, y_train_pred, id_train = model_train(ds, "models/%s.hd5" % (1, ), 59, 'D')
# visualize_model_prediction(ds, y_train_pred, id_train[:y_train_pred.shape[0]], y_test_pred, id_test)
#
# metrics_nwrmse(ds.set_index('obs_id').loc[id_test, 'Consumption'].values, y_test_pred, ds.set_index('obs_id').loc[id_test, 'ForecastId'].values)