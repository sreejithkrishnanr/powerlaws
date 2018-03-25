HPARAMS = {
    'gb': {
        'DEFAULT': {
            'max_depth': 3,
            'n_estimators': 100,
            'learning_rate': 0.01,
        },
        'freq_D': {
            'learning_rate': 0.11,
            'max_depth': 3,
            'n_estimators': 100,
            'colsample_bytree': 1,
            'colsample_bylevel': 0.9,
            'gamma': 0,
            'reg_alpha': 1.1,
            'reg_lambda': 1
        },
        'freq_h': {
            'colsample_bylevel': 0.7,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'learning_rate': 0.1,
            'max_depth': 4,
            'n_estimators': 75,
            'reg_alpha': 1.2,
            'reg_lambda': 1.2
        },
        'freq_900s': {
            'colsample_bylevel': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'learning_rate': 0.11,
            'max_depth': 5,
            'n_estimators': 20,
            'reg_alpha': 1.4,
            'reg_lambda': 1.2
        }
    },
    'recursive_gb': {
        'DEFAULT': {
            'max_depth': 3,
            'n_estimators': 100,
            'learning_rate': 0.01,
        },
        'freq_D': {
            'learning_rate': 0.11,
            'max_depth': 3,
            'n_estimators': 100,
            'colsample_bytree': 1,
            'colsample_bylevel': 0.9,
            'gamma': 0,
            'reg_alpha': 1.1,
            'reg_lambda': 1,
            'input_window_size': 14
        },
        'freq_h': {
            'colsample_bylevel': 0.7,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'learning_rate': 0.1,
            'max_depth': 4,
            'n_estimators': 75,
            'reg_alpha': 1.2,
            'reg_lambda': 1.2,
            'input_window_size': 24*2
        },
        'freq_900s': {
            'colsample_bylevel': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'learning_rate': 0.11,
            'max_depth': 5,
            'n_estimators': 20,
            'reg_alpha': 1.4,
            'reg_lambda': 1.2,
            'input_window_size': 24*4*2
        }
    },
    'gb_stat': {
        'DEFAULT': {
            'gamma': 0,
        },
        'freq_D': {
            'learning_rate': 0.11,
            'max_depth': 3,
            'n_estimators': 100,
            'colsample_bytree': 1,
            'colsample_bylevel': 0.9,
            'reg_alpha': 1.1,
            'reg_lambda': 1,
            'trend_column': 'ConsumptionWeeklyMean'
        },
        'freq_h': {
            'colsample_bylevel': 0.7,
            'colsample_bytree': 0.8,
            'learning_rate': 0.1,
            'max_depth': 4,
            'n_estimators': 75,
            'reg_alpha': 1.2,
            'reg_lambda': 1.2,
            'trend_column': 'ConsumptionDailyMean'
        },
        'freq_900s': {
            'colsample_bylevel': 0.8,
            'colsample_bytree': 0.8,
            'learning_rate': 0.11,
            'max_depth': 5,
            'n_estimators': 20,
            'reg_alpha': 1.4,
            'reg_lambda': 1.2,
            'trend_column': 'ConsumptionDailyMean'
        }
    },
    'rnn': {

    }
}


def _merge_params(*params):
    result = {}

    for param in params:
        for key, val in param.items():
            result[key] = val

    return result


def get_hparams(site_id, frequency, model, hparams=HPARAMS):
    model_params = hparams[model]

    default_params = model_params['DEFAULT']
    freq_params = model_params['freq_%s' % (frequency, )]

    site_key = "site_%s" % (site_id,) if site_id is not None else None
    site_params = model_params[site_key] if site_key is not None and site_key in model_params.keys() else {}

    return _merge_params(default_params, freq_params, site_params)
