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


def gb_evaluate_model(x_train, y_train, x_test, y_test, site_id, frequency, verbose=False, **kwargs):
    hparams = get_hparams(site_id, frequency, 'gb')
    regressor = _build_regressor_from_params(hparams)

    regressor.fit(x_train, y_train, eval_set=[(x_test, y_test)], eval_metric='rmse', verbose=verbose)

    return np.maximum(regressor.predict(x_test), 0), None


def gb_build_model(x, y, site_id, frequency, output_path, verbose=False, **kwargs):
    regressor = _build_regressor_from_params(get_hparams(site_id, frequency, 'gb'))
    regressor.fit(x, y, verbose=verbose)

    with open(output_path, 'wb') as f:
        pickle.dump(regressor, f, protocol=4)


def gb_predict_model(x_test, model_path, **kwargs):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    return np.maximum(model.predict(x_test), 0)
