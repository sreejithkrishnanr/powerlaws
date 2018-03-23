import numpy as np
import pandas as pd

from src.utils.constants import get_output_window_size
from src.models.model_gb import gb_evaluate_model, gb_build_model, gb_predict_model
from src.models.model_rnn import rnn_evaluate_model, rnn_build_model, rnn_predict_model

MODEL_REGISTRY = {
    'gb': [gb_evaluate_model, gb_build_model, gb_predict_model],
    'rnn': [rnn_evaluate_model, rnn_build_model, rnn_predict_model]
}


def has_good_weather_forecast(dataset):
    return np.sum(dataset['HasTemperature'].values) / dataset.shape[0] > 0.6


def select_features(dataset, frequency, novalue=False, train_data=None, include_site_id=False,
                    use_consumption_per_sa=False):

    if novalue and train_data is None:
        raise Exception('Train data is required for selecting features for test data')

    if use_consumption_per_sa:
        y_val_column = 'ConsumptionPerSurfaceArea'
    else:
        y_val_column = 'Consumption'

    if use_consumption_per_sa:
        y_dep_features = [
            'ConsumptionDailyMeanPerSurfaceArea',
            'ConsumptionWeeklyMeanPerSurfaceArea',
            'ConsumptionBiWeeklyMeanPerSurfaceArea',
            'ConsumptionMonthlyMeanPerSurfaceArea',
        ]
    else:
        y_dep_features = [
            'ConsumptionDailyMean',
            'ConsumptionWeeklyMean',
            'ConsumptionBiWeeklyMean',
            'ConsumptionMonthlyMean',
        ]

    weather_forecast_features = [
        'PotentialMeanHeating', 'PotentialMeanCooling', 'PotentialMinHeating', 'PotentialMaxHeating',
        'PotentialMinCooling', 'PotentialMaxCooling', 'PotentialDailyMeanHeating', 'PotentialDailyMeanCooling',
        'PotentialWeeklyMeanHeating', 'PotentialWeeklyMeanCooling', 'PotentialMonthlyMeanHeating',
        'PotentialMonthlyMeanCooling', 'PotentialQuarterlyMeanHeating', 'PotentialQuarterlyMeanCooling',
        'PotentialYearlyMeanHeating', 'PotentialYearlyMeanCooling',
        'DistanceMean', 'DistanceVariance', 'NumStations', 'HasTemperature', 'TemperatureVariance',
        # 'TemperatureMeanDiff', 'TemperatureMinDiff', 'TemperatureMaxDiff'
    ]

    features = [
        'IsLeapYear', 'IsMonthEnd', 'IsMonthStart', 'IsQuarterEnd',
        'IsQuarterStart', 'IsYearEnd', 'IsYearStart', 'DayOfMonth_cos', 'DayOfMonth_sin',
        'DayOfWeek_cos', 'DayOfWeek_sin', 'DayOfYear_cos',
        'DayOfYear_sin', 'Hour_cos', 'Hour_sin', 'Minute_cos', 'Minute_sin',
        'Month_cos', 'Month_sin', 'Quarter_cos', 'Quarter_sin',
        'WeekOfYear_cos', 'WeekOfYear_sin', 'IsSpecialHoliday', 'IsWeekend',
        'IsHoliday', y_val_column, 'ForecastId', 'SiteId'
    ] + y_dep_features

    if frequency == 'D':
        features.remove('Hour_cos')
        features.remove('Hour_sin')
        features.remove('Minute_cos')
        features.remove('Minute_sin')
    elif frequency == 'h':
        features.remove('Minute_cos')
        features.remove('Minute_sin')
    elif frequency == '900s':
        pass
    else:
        raise Exception('Unknown frequency %s' % (frequency,))

    if has_good_weather_forecast(train_data if novalue else dataset):
        features += weather_forecast_features

    if novalue:
        features.remove(y_val_column)

    if not include_site_id:
        features.remove('SiteId')

    feature_set = dataset.set_index('obs_id')[features]

    output_window_size = get_output_window_size(frequency)

    if not novalue:
        for feature in y_dep_features:
            feature_set[feature] = feature_set.groupby('ForecastId')[feature].shift(output_window_size+1).values
            feature_set[feature] = feature_set.groupby('ForecastId')[feature]\
                .transform(lambda x: x.fillna(x[x.first_valid_index()])).values

    if include_site_id:
        feature_set = pd.get_dummies(feature_set, columns=['SiteId'])

    x = feature_set.drop(columns=['ForecastId'] + ([y_val_column] if not novalue else []))
    y = feature_set[y_val_column] if not novalue else None
    groups = feature_set['ForecastId']

    return x, y, groups
