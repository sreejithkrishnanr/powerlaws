import tempfile
import zipfile
import os
import shutil

import numpy as np

from keras.layers import Dense, TimeDistributed, GRU
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler
import joblib

from src.models.model_common import get_output_window_size


def _generate_train_forecast_ts(x, y, output_window_size, pad=True, stride=1):
    assert x.shape[0] == y.shape[0]

    min_size = output_window_size
    input_size = x.shape[0]

    if input_size < min_size and pad:
        num_pads = min_size - input_size

        x_pad = np.zeros((num_pads, x.shape[1]))
        y_pad = np.zeros((num_pads, y.shape[1]))
        x = np.vstack((x, x_pad))
        y = np.vstack((y, y_pad))

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
    for fid in ids:
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
    model.add(GRU(units=25, return_sequences=True, batch_input_shape=(None, x_shape[1], x_shape[2]),
                  recurrent_dropout=0.4, dropout=0.1))
    # model.add(GRU(units=25, return_sequences=True, recurrent_dropout=0.4, dropout=0.1))
    model.add(TimeDistributed(Dense(units=1, activation='linear')))
    model.compile(loss='mse', optimizer='adam')

    return model


def rnn_evaluate_model(x_train, y_train, g_train, x_test, y_test, g_test, site_id, frequency, verbose=False, **kwargs):
    output_window_size = get_output_window_size(frequency)

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    ts_x_train, ts_y_train = _generate_train_ts(
        scaler_x.fit_transform(x_train.values),
        scaler_y.fit_transform(y_train.values.reshape(-1, 1)),
        g_train.values,
        output_window_size
    )

    ts_x_test, ts_y_test = _generate_train_ts(
        scaler_x.transform(x_test.values),
        scaler_y.transform(y_test.values.reshape(-1, 1)),
        g_test.values,
        output_window_size,
        stride=output_window_size
    )

    model = _model_build_lstm(ts_x_train.shape, ts_y_train.shape)
    history = model.fit(ts_x_train, ts_y_train, batch_size=50, callbacks=[
        EarlyStopping(patience=10)
    ], validation_data=(ts_x_test, ts_y_test), epochs=400, verbose=0)

    return scaler_y.inverse_transform(model.predict(ts_x_test).ravel()), {'epochs': history.epoch[-1]}


def rnn_build_model(x, y, groups, site_id, frequency, output_path, evaluated_args, verbose=False, **kwargs):
    output_window_size = get_output_window_size(frequency)

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    ts_x_train, ts_y_train = _generate_train_ts(
        scaler_x.fit_transform(x.values),
        scaler_y.fit_transform(y.values.reshape(-1, 1)),
        groups.values,
        output_window_size
    )

    model = _model_build_lstm(ts_x_train.shape, ts_y_train.shape)
    model.fit(ts_x_train, ts_y_train, epochs=evaluated_args['epochs'], verbose=1 if verbose else 0)

    temp_dir = tempfile.mkdtemp()

    rnn_path = os.path.join(temp_dir, 'rnn.hd5')
    model.save(rnn_path)

    scaler_x_path = os.path.join(temp_dir, 'scaler_x.pkl')
    joblib.dump(scaler_x, scaler_x_path)

    scaler_y_path = os.path.join(temp_dir, 'scaler_y.pkl')
    joblib.dump(scaler_y, scaler_y_path)

    with zipfile.ZipFile(output_path, mode='w') as f:
        f.write(rnn_path, 'rnn.hd5')
        f.write(scaler_x_path, 'scaler_x.pkl')
        f.write(scaler_y_path, 'scaler_y.pkl')

    shutil.rmtree(temp_dir)


def rnn_predict_model(x_test, g_test, model_path, frequency, **kwargs):
    output_window_size = get_output_window_size(frequency)

    temp_dir = tempfile.mkdtemp()

    rnn_path = os.path.join(temp_dir, 'rnn.hd5')
    scaler_x_path = os.path.join(temp_dir, 'scaler_x.pkl')
    scaler_y_path = os.path.join(temp_dir, 'scaler_y.pkl')

    with zipfile.ZipFile(model_path, 'r') as f:
        f.extract('rnn.hd5', temp_dir)
        f.extract('scaler_x.pkl', temp_dir)
        f.extract('scaler_y.pkl', temp_dir)

    rnn = load_model(rnn_path)
    scaler_x = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)

    x, _ = _generate_train_ts(
        scaler_x.transform(x_test.values), np.zeros(x_test.shape[0]).reshape(-1, 1), g_test.values, output_window_size)

    y_pred = scaler_y.inverse_transform(rnn.predict(x).ravel())

    shutil.rmtree(temp_dir)

    return y_pred[:x_test.shape[0]]
