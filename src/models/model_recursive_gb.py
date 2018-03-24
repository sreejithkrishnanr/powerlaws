import pickle

import numpy as np

from xgboost import XGBRegressor

from src.models.model_hparams import get_hparams


def _build_regressor_from_params(params):
    max_depth = params['max_depth']
    n_estimators = params['n_estimators']
    learning_rate = params['learning_rate']
    colsample_bytree = params['colsample_bytree']
    colsample_bylevel = params['colsample_bylevel']
    gamma = params['gamma']
    reg_alpha = params['reg_alpha']
    reg_lambda = params['reg_lambda']

    return XGBRegressor(max_depth=max_depth, n_estimators=n_estimators, colsample_bytree=colsample_bytree,
                        colsample_bylevel=colsample_bylevel, gamma=gamma, reg_alpha=reg_alpha, reg_lambda=reg_lambda,
                        learning_rate=learning_rate, n_jobs=8)


def _generate_train_forecast_ts(x, y, input_window_size, output_window_size, pad=True):
    assert x.shape[0] == y.shape[0]

    min_size = input_window_size + output_window_size
    input_size = x.shape[0]

    if input_size < min_size and pad:
        num_pads = min_size - input_size

        x_pad = np.zeros((num_pads, x.shape[1]))
        y_pad = np.zeros((num_pads, y.shape[1]))
        x = np.vstack((x_pad, x))
        y = np.vstack((y_pad, y))

    assert x.shape[0] >= min_size

    num_samples = x.shape[0] - input_window_size - output_window_size + 1

    x_res = []
    y_res = []

    for i in range(num_samples):
        x_t = np.hstack((x[i:i + input_window_size], y[i:i + input_window_size].reshape(-1, 1)))

        x_t_plus_1 = x[i + input_window_size:i + input_window_size + output_window_size]
        y_t_plus_1 = y[i + input_window_size:i + input_window_size + output_window_size]

        x_res.append(np.concatenate((x_t.ravel(), x_t_plus_1.ravel())))
        y_res.append(y_t_plus_1)

    return np.array(x_res), np.array(y_res)


def _generate_train_ts(x, y, forecast_ids, input_window_size, output_window_size):
    assert x.shape[0] == y.shape[0]
    assert y.shape[0] == forecast_ids.shape[0]

    ids = np.unique(forecast_ids)

    agg_x = []
    agg_y = []
    for fid in ids:
        fx = x[forecast_ids == fid, :]
        fy = y[forecast_ids == fid, :]

        fx_res, fy_res = _generate_train_forecast_ts(fx, fy, input_window_size, output_window_size)

        agg_x.append(fx_res)
        agg_y.append(fy_res)

    agg_x = np.concatenate(agg_x)
    agg_y = np.concatenate(agg_y)

    return agg_x, agg_y.ravel()


def _make_forecast_predictions(model, x_test, ts_test, x_train, y_train, ts_train, input_window_size, frequency):
    if frequency == 'D':
        td = np.timedelta64(1, 'D')
    elif frequency == 'h':
        td = np.timedelta64(1, 'h')
    elif frequency == '900s':
        td = np.timedelta64(900, 's')
    else:
        raise Exception("Unsupported frequency %s" % (frequency, ))

    start_time = ts_test.min()

    train_window_start_time = start_time - (input_window_size * td)
    train_window_end_time = start_time

    train_window = (ts_train >= train_window_start_time) & (ts_train < train_window_end_time)

    assert np.sum(train_window) == input_window_size

    x_t_minus_1 = np.hstack((x_train.loc[train_window, :].values, y_train.loc[train_window].values.reshape((-1, 1))))

    y_pred = []
    for i in range(x_test.shape[0]):
        x_t = x_test.iloc[i, :]
        x = np.concatenate((x_t_minus_1.ravel(), x_t)).reshape((1, -1))
        y = model.predict(x)
        y_pred.append(y)

        x_t_minus_1 = np.delete(x_t_minus_1, (0,), axis=0)
        x_t_minus_1 = np.vstack((x_t_minus_1, np.hstack((x_t, y))))

    return np.array(y_pred)


def _make_predictions(model, x_test, g_test, ts_test, x_train, y_train, ts_train, input_window_size, frequency):
    fids = np.unique(g_test)

    y_pred = []
    for fid in fids:
        forecast_x_test = x_test.loc[g_test == fid, :]
        forecast_ts_test = ts_test.loc[g_test == fid]
        forecast_y_pred = _make_forecast_predictions(
            model, forecast_x_test, forecast_ts_test, x_train, y_train, ts_train, input_window_size, frequency)

        y_pred.append(forecast_y_pred)

    return np.concatenate(y_pred).ravel()


def recursive_gb_evaluate_model(x_train, y_train, g_train, ts_train, x_test, g_test, ts_test, site_id,
                                frequency, verbose=False, **kwargs):
    hparams = get_hparams(site_id, frequency, 'recursive_gb')
    regressor = _build_regressor_from_params(hparams)
    input_window_size = hparams['input_window_size']

    tsx_train, tsy_train = _generate_train_ts(x_train.values, y_train.values.reshape((-1, 1)), g_train.values, input_window_size, 1)

    regressor.fit(tsx_train, tsy_train, verbose=verbose)

    return _make_predictions(
        regressor, x_test, g_test, ts_test, x_train, y_train, ts_train, input_window_size, frequency), None


def recursive_gb_build_model(x, y, groups, site_id, frequency, output_path, verbose=False, **kwargs):
    hparams = get_hparams(site_id, frequency, 'recursive_gb')
    regressor = _build_regressor_from_params(hparams)

    tsx, tsy = _generate_train_ts(x.values, y.values.reshape((-1, 1)), groups.values, hparams['input_window_size'], 1)
    regressor.fit(tsx, tsy, verbose=verbose)

    with open(output_path, 'wb') as f:
        pickle.dump(regressor, f, protocol=4)


def recursive_gb_predict_model(model_path, x_test, g_test, ts_test, site_id, x_train,
                               y_train, ts_train, frequency, **kwargs):
    h_params = get_hparams(site_id, frequency, 'recursive_gb')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    return _make_predictions(
        model, x_test, g_test, ts_test, x_train, y_train, ts_train, h_params['input_window_size'], frequency)
