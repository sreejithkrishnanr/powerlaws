import numpy as np
import pandas as pd

from src.utils.constants import get_output_window_size
from src.models.model_gb import gb_evaluate_model, gb_build_model, gb_predict_model
from src.models.model_rnn import rnn_evaluate_model, rnn_build_model, rnn_predict_model
from src.models.model_gb_recursive import recursive_gb_evaluate_model, recursive_gb_build_model, recursive_gb_predict_model
from src.models.model_gb_stationary import gb_stat_evaluate_model, gb_stat_build_model, gb_stat_predict_model
from src.models.model_gb_log import gb_log_evaluate_model, gb_log_build_model, gb_log_predict_model

MODEL_REGISTRY = {
    'gb': [gb_evaluate_model, gb_build_model, gb_predict_model],
    'gb_stat': [gb_stat_evaluate_model, gb_stat_build_model, gb_stat_predict_model],
    'rnn': [rnn_evaluate_model, rnn_build_model, rnn_predict_model],
    'gb_recursive': [recursive_gb_evaluate_model, recursive_gb_build_model, recursive_gb_predict_model],
    'gb_log': [gb_log_evaluate_model, gb_log_build_model, gb_log_predict_model]
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

    y_dep_features = []
    if frequency == 'D' or frequency == 'h' or frequency == '900s':
        y_dep_features += ['ConsumptionDailyMean', 'ConsumptionWeeklyMean']
    if frequency == 'D' or frequency == 'h':
        y_dep_features += ['ConsumptionBiWeeklyMean', 'ConsumptionMonthlyMean']
    if frequency == 'h' or frequency == '900s':
        y_dep_features += ['ConsumptionHalfDayMean', 'ConsumptionQuarterDayMean']
    if frequency == '900s':
        y_dep_features += ['ConsumptionHourlyMean']

    if use_consumption_per_sa:
        y_dep_features = list(map(lambda v: "%sPerSurfaceArea" % (v, ), y_dep_features))

    weather_forecast_features = [
        'PotentialMeanHeating', 'PotentialMeanCooling', 'PotentialMinHeating', 'PotentialMaxHeating',
        'PotentialMinCooling', 'PotentialMaxCooling',
        'PotentialWeeklyMeanHeating', 'PotentialWeeklyMeanCooling', 'PotentialMonthlyMeanHeating',
        'PotentialMonthlyMeanCooling', 'PotentialQuarterlyMeanHeating', 'PotentialQuarterlyMeanCooling',
        'PotentialYearlyMeanHeating', 'PotentialYearlyMeanCooling',
        'DistanceMean', 'DistanceVariance', 'NumStations', 'HasTemperature', 'TemperatureVariance',
        # 'TemperatureMeanDiff', 'TemperatureMinDiff', 'TemperatureMaxDiff'
    ]

    if frequency == 'h' or frequency == '900s':
        weather_forecast_features += [
            'PotentialDailyMeanHeating', 'PotentialDailyMeanCooling',
            'PotentialHalfDayMeanHeating', 'PotentialHalfDayMeanCooling',
            'PotentialQuarterDayMeanHeating', 'PotentialQuarterDayMeanCooling',
        ]

    if frequency == '900s':
        weather_forecast_features += [
            'PotentialBiHourlyMeanHeating', 'PotentialBiHourlyMeanCooling',
            'PotentialHourlyMeanHeating', 'PotentialHourlyMeanCooling',
            'PotentialHalfHourlyMeanHeating', 'PotentialHalfHourlyMeanCooling'
        ]

    holiday_features = [
        'SpecialHoliday', 'Weekend', 'Holiday',
        'SpecialHolidayRemainingYearCount',
        'SpecialHolidayRemainingQuarterCount',
        'SpecialHolidayRemainingMonthCount',
        'WeekendRemainingYearCount',
        'WeekendRemainingQuarterCount',
        'WeekendRemainingMonthCount',
        'HolidayRemainingYearCount',
        'HolidayRemainingQuarterCount',
        'HolidayRemainingMonthCount', 'NumSpecialHolidayWithin2',
        'NumWeekendWithin2', 'NumHolidayWithin2', 'NumSpecialHolidayWithin3',
        'NumWeekendWithin3', 'NumHolidayWithin3', 'NumSpecialHolidayWithin4',
        'NumWeekendWithin4', 'NumHolidayWithin4', 'NumSpecialHolidayWithin5',
        'NumWeekendWithin5', 'NumHolidayWithin5', 'NumSpecialHolidayWithin6',
        'NumWeekendWithin6', 'NumHolidayWithin6', 'NumSpecialHolidayWithin7',
        'NumWeekendWithin7', 'NumHolidayWithin7', 'NumSpecialHolidayWithin10',
        'NumWeekendWithin10', 'NumHolidayWithin10', 'NumSpecialHolidayWithin15',
        'NumWeekendWithin15', 'NumHolidayWithin15', 'NumSpecialHolidayWithin30',
        'NumWeekendWithin30', 'NumHolidayWithin30',
        'NumSpecialHolidayWithin120', 'NumWeekendWithin120',
        'NumHolidayWithin120',
    ]

    features = [
        'IsLeapYear', 'IsMonthEnd', 'IsMonthStart', 'IsQuarterEnd',
        'IsQuarterStart', 'IsYearEnd', 'IsYearStart', 'DayOfMonth_cos', 'DayOfMonth_sin',
        'DayOfWeek_cos', 'DayOfWeek_sin', 'DayOfYear_cos',
        'DayOfYear_sin', 'Hour_cos', 'Hour_sin', 'Minute_cos', 'Minute_sin',
        'Month_cos', 'Month_sin', 'Quarter_cos', 'Quarter_sin',
        'WeekOfYear_cos', 'WeekOfYear_sin', y_val_column, 'ForecastId', 'SiteId', 'Timestamp'
    ] + y_dep_features + holiday_features

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

    x = feature_set.drop(columns=['ForecastId', 'Timestamp'] + ([y_val_column] if not novalue else []))
    y = feature_set[y_val_column] if not novalue else None
    groups = feature_set['ForecastId']
    ts = feature_set['Timestamp']

    return x, y, groups, ts
