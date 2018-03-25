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


def gb_stat_evaluate_model(x_train, y_train, x_test, y_test, site_id, frequency, verbose=False, **kwargs):
    hparams = get_hparams(site_id, frequency, 'gb_stat')
    regressor = _build_regressor_from_params(hparams)

    trend_column = hparams['trend_column']
    y_train = y_train - x_train[trend_column]
    regressor.fit(x_train, y_train, verbose=verbose)

    return np.maximum(x_test[trend_column].values + regressor.predict(x_test), 0), None


def gb_stat_build_model(x, y, site_id, frequency, output_path, verbose=False, **kwargs):
    hparams = get_hparams(site_id, frequency, 'gb_stat')
    regressor = _build_regressor_from_params(hparams)

    trend_column = hparams['trend_column']
    y = y - x[trend_column]
    regressor.fit(x, y, verbose=verbose)

    with open(output_path, 'wb') as f:
        pickle.dump(regressor, f, protocol=4)


def gb_stat_predict_model(x_test, model_path, site_id, frequency, **kwargs):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    hparams = get_hparams(site_id, frequency, 'gb_stat')
    trend_column = hparams['trend_column']

    return np.maximum(x_test[trend_column].values + model.predict(x_test), 0)
