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

        },
        'freq_900s': {

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
