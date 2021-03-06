import logging
import os
import sys
from functools import reduce

import click
import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm


def build_holidays(dataset, holidays, **kwargs):
    dataset = pd.merge_asof(
        dataset,
        holidays.drop(columns=['DayOfWeek', 'WeekOfYear', 'Year', 'Month', 'Quarter', 'SiteId']),
        left_on='Date', right_on='Date', tolerance=pd.Timedelta('1h'), direction='nearest'
    )

    return dataset


def build_site_metadata(dataset, metadata, frequency, **kwargs):
    dataset['SamplingRate'] = metadata['Sampling']
    dataset['BaseTemperature'] = metadata['BaseTemperature']
    dataset['SurfaceArea'] = metadata['Surface']
    dataset['TemperatureMeanDiff'] = dataset['TemperatureMean'] - dataset['BaseTemperature']
    dataset['TemperatureMinDiff'] = dataset['TemperatureMin'] - dataset['BaseTemperature']
    dataset['TemperatureMaxDiff'] = dataset['TemperatureMax'] - dataset['BaseTemperature']
    dataset['TemperatureWeeklyMeanDiff'] = dataset['TemperatureWeeklyMean'] - dataset['BaseTemperature']
    dataset['TemperatureMonthlyMeanDiff'] = dataset['TemperatureMonthlyMean'] - dataset['BaseTemperature']
    dataset['TemperatureQuarterlyMeanDiff'] = dataset['TemperatureQuarterlyMean'] - dataset['BaseTemperature']
    dataset['TemperatureYearlyMeanDiff'] = dataset['TemperatureYearlyMean'] - dataset['BaseTemperature']

    dataset['PotentialMeanHeating'] = np.maximum(dataset['TemperatureMeanDiff'], 0)
    dataset['PotentialMeanCooling'] = np.abs(np.minimum(dataset['TemperatureMeanDiff'], 0))
    dataset['PotentialMinHeating'] = np.minimum(dataset['TemperatureMinDiff'], 0)
    dataset['PotentialMaxHeating'] = np.maximum(dataset['TemperatureMaxDiff'], 0)
    dataset['PotentialMinCooling'] = np.abs(np.minimum(dataset['TemperatureMinDiff'], 0))
    dataset['PotentialMaxCooling'] = np.abs(np.minimum(dataset['TemperatureMaxDiff'], 0))
    dataset['PotentialWeeklyMeanHeating'] = np.maximum(dataset['TemperatureWeeklyMeanDiff'], 0)
    dataset['PotentialWeeklyMeanCooling'] = np.abs(np.minimum(dataset['TemperatureWeeklyMeanDiff'], 0))
    dataset['PotentialMonthlyMeanHeating'] = np.maximum(dataset['TemperatureMonthlyMeanDiff'], 0)
    dataset['PotentialMonthlyMeanCooling'] = np.abs(np.minimum(dataset['TemperatureMonthlyMeanDiff'], 0))
    dataset['PotentialQuarterlyMeanHeating'] = np.maximum(dataset['TemperatureQuarterlyMeanDiff'], 0)
    dataset['PotentialQuarterlyMeanCooling'] = np.abs(np.minimum(dataset['TemperatureQuarterlyMeanDiff'], 0))
    dataset['PotentialYearlyMeanHeating'] = np.maximum(dataset['TemperatureYearlyMeanDiff'], 0)
    dataset['PotentialYearlyMeanCooling'] = np.abs(np.minimum(dataset['TemperatureYearlyMeanDiff'], 0))

    if frequency == 'h' or frequency == '900s':
        dataset['TemperatureDailyMeanDiff'] = dataset['TemperatureDailyMean'] - dataset['BaseTemperature']
        dataset['PotentialDailyMeanHeating'] = np.maximum(dataset['TemperatureDailyMeanDiff'], 0)
        dataset['PotentialDailyMeanCooling'] = np.abs(np.minimum(dataset['TemperatureDailyMeanDiff'], 0))

        dataset['TemperatureHalfDayMeanDiff'] = dataset['TemperatureHalfDayMean'] - dataset['BaseTemperature']
        dataset['PotentialHalfDayMeanHeating'] = np.maximum(dataset['TemperatureHalfDayMeanDiff'], 0)
        dataset['PotentialHalfDayMeanCooling'] = np.abs(np.minimum(dataset['TemperatureHalfDayMeanDiff'], 0))

        dataset['TemperatureQuarterDayMeanDiff'] = dataset['TemperatureQuarterDayMean'] - dataset['BaseTemperature']
        dataset['PotentialQuarterDayMeanHeating'] = np.maximum(dataset['TemperatureQuarterDayMeanDiff'], 0)
        dataset['PotentialQuarterDayMeanCooling'] = np.abs(np.minimum(dataset['TemperatureQuarterDayMeanDiff'], 0))

    if frequency == '900s':
        dataset['TemperatureBiHourlyMeanDiff'] = dataset['TemperatureBiHourlyMean'] - dataset['BaseTemperature']
        dataset['PotentialBiHourlyMeanHeating'] = np.maximum(dataset['TemperatureBiHourlyMeanDiff'], 0)
        dataset['PotentialBiHourlyMeanCooling'] = np.abs(np.minimum(dataset['TemperatureBiHourlyMeanDiff'], 0))

        dataset['TemperatureHourlyMeanDiff'] = dataset['TemperatureHourlyMean'] - dataset['BaseTemperature']
        dataset['PotentialHourlyMeanHeating'] = np.maximum(dataset['TemperatureHourlyMeanDiff'], 0)
        dataset['PotentialHourlyMeanCooling'] = np.abs(np.minimum(dataset['TemperatureHourlyMeanDiff'], 0))

        dataset['TemperatureHalfHourlyMeanDiff'] = dataset['TemperatureHalfHourlyMean'] - dataset['BaseTemperature']
        dataset['PotentialHalfHourlyMeanHeating'] = np.maximum(dataset['TemperatureHalfHourlyMeanDiff'], 0)
        dataset['PotentialHalfHourlyMeanCooling'] = np.abs(np.minimum(dataset['TemperatureHalfHourlyMeanDiff'], 0))

    return dataset


def build_timestamp(dataset, **kwargs):
    dt = dataset['Timestamp'].dt
    dataset['Date'] = dt.ceil('D')
    dataset['DayOfMonth'] = dt.day
    dataset['DayOfWeek'] = dt.dayofweek
    dataset['DayOfYear'] = dt.dayofyear
    dataset['DaysInMonth'] = dt.days_in_month
    dataset['Hour'] = dt.hour
    dataset['IsLeapYear'] = dt.is_leap_year
    dataset['IsMonthEnd'] = dt.is_month_end
    dataset['IsMonthStart'] = dt.is_month_start
    dataset['IsQuarterEnd'] = dt.is_quarter_end
    dataset['IsQuarterStart'] = dt.is_quarter_start
    dataset['IsYearEnd'] = dt.is_year_end
    dataset['IsYearStart'] = dt.is_year_start
    dataset['Minute'] = dt.minute
    dataset['Month'] = dt.month
    dataset['Quarter'] = dt.quarter
    dataset['WeekOfYear'] = dt.weekofyear

    dataset['DayOfMonth_cos'] = np.cos(2 * np.pi * dataset['DayOfMonth']/dataset['DaysInMonth'])
    dataset['DayOfMonth_sin'] = np.sin(2 * np.pi * dataset['DayOfMonth'] / dataset['DaysInMonth'])

    dataset['DayOfWeek_cos'] = np.cos(2 * np.pi * (dataset['DayOfWeek'] + 1.0 )/ 7.0)
    dataset['DayOfWeek_sin'] = np.sin(2 * np.pi * (dataset['DayOfWeek'] + 1.0) / 7.0)

    dataset['DaysInYear'] = 365
    dataset.loc[dataset['IsLeapYear'] == True, 'DaysInYear'] = 366
    dataset['DayOfYear_cos'] = np.cos(2 * np.pi * dataset['DayOfYear']/dataset['DaysInYear'])
    dataset['DayOfYear_sin'] = np.sin(2 * np.pi * dataset['DayOfYear'] / dataset['DaysInYear'])

    dataset['Hour_cos'] = np.cos(2 * np.pi * (dataset['Hour'] + 1.0) / 23.0)
    dataset['Hour_sin'] = np.sin(2 * np.pi * (dataset['Hour'] + 1.0) / 23.0)

    dataset['Minute_cos'] = np.cos(2 * np.pi * (dataset['Hour'] + 1.0) / 59.0)
    dataset['Minute_sin'] = np.sin(2 * np.pi * (dataset['Hour'] + 1.0) / 59.0)

    dataset['Month_cos'] = np.cos(2 * np.pi * (dataset['Month']) / 12.0)
    dataset['Month_sin'] = np.sin(2 * np.pi * (dataset['Month']) / 12.0)

    dataset['Quarter_cos'] = np.cos(2 * np.pi * (dataset['Quarter']) / 4.0)
    dataset['Quarter_sin'] = np.sin(2 * np.pi * (dataset['Quarter']) / 4.0)

    dataset['WeekOfYear_cos'] = np.cos(2 * np.pi * (dataset['Quarter']) / 53.0)
    dataset['WeekOfYear_sin'] = np.sin(2 * np.pi * (dataset['Quarter']) / 53.0)

    return dataset


def build_temperature(dataset, weather, frequency, **kwargs):
    logger = logging.getLogger(__name__)

    dataset = dataset[:]

    ds = weather.set_index('Timestamp')

    weather = ds.resample(frequency).agg({
        'DistanceMean': 'mean',
        'DistanceMin': 'min',
        'DistanceMax': 'max',
        'DistanceVariance': 'mean',
        'NumStations': 'mean',
        'TemperatureMean': 'mean',
        'TemperatureVariance': 'mean',
        'TemperatureMin': 'min',
        'TemperatureMax': 'max'
    })

    if frequency == 'D':
        limit = 2
        freq = 1
    elif frequency == 'h':
        limit = 6
        freq = 24
    elif frequency == '900s':
        limit = 4*6
        freq = 4*24
    else:
        raise Exception("Unknown frequency %s" % (frequency, ))

    weather['TemperatureWeeklyMean'] = weather['TemperatureMean'].rolling(window=freq * 7).mean()
    weather['TemperatureMonthlyMean'] = weather['TemperatureMean'].rolling(window=freq * 30).mean()
    weather['TemperatureQuarterlyMean'] = weather['TemperatureMean'].rolling(window=freq * 30 * 4).mean()
    weather['TemperatureYearlyMean'] = weather['TemperatureMean'].rolling(window=freq * 30 * 12).mean()

    weather_agg_columns = [
        'TemperatureWeeklyMean',
        'TemperatureMonthlyMean',
        'TemperatureQuarterlyMean',
        'TemperatureYearlyMean'
    ]

    if frequency == 'h' or frequency == '900s':
        weather['TemperatureDailyMean'] = weather['TemperatureMean'].rolling(window=freq).mean()
        weather['TemperatureHalfDayMean'] = weather['TemperatureMean'].rolling(window=freq // 2).mean()
        weather['TemperatureQuarterDayMean'] = weather['TemperatureMean'].rolling(window=freq // 4).mean()
        weather_agg_columns += ['TemperatureDailyMean', 'TemperatureHalfDayMean', 'TemperatureQuarterDayMean']

    if frequency == '900s':
        weather['TemperatureBiHourlyMean'] = weather['TemperatureMean'].rolling(window=freq // 12).mean()
        weather['TemperatureHourlyMean'] = weather['TemperatureMean'].rolling(window=freq // 24).mean()
        weather['TemperatureHalfHourlyMean'] = weather['TemperatureMean'].rolling(window=freq // 48).mean()
        weather_agg_columns += ['TemperatureBiHourlyMean', 'TemperatureHourlyMean', 'TemperatureHalfHourlyMean']

    for column in weather_agg_columns:
        first_valid_index = weather[column].first_valid_index()
        if first_valid_index:
            weather[column].fillna(weather[column][weather[column].first_valid_index()], inplace=True)

    weather = weather.interpolate(method='linear', limit=limit, limit_direction='both')

    if frequency == 'D':
        tolerance = '2D'
    elif frequency == 'h':
        tolerance = '2h'
    elif frequency == '900s':
        tolerance = '1800s'
    else:
        raise Exception("Unknown frequency %s" % (frequency, ))

    weather = weather.reset_index()
    dataset = pd.merge_asof(dataset, weather, left_on='Timestamp', right_on='Timestamp',
                            tolerance=pd.Timedelta(tolerance), direction='nearest')

    dataset['HasTemperature'] = np.logical_not(np.isnan(dataset['TemperatureMin'].values))
    dataset[weather.drop(columns=['Timestamp']).keys()] = dataset[weather.drop(columns=['Timestamp']).keys()].fillna(0)

    return dataset


def build_consumption_value(dataset, frequency, **kwargs):
    d1 = dataset.set_index('Timestamp').groupby(['ForecastId', pd.Grouper(freq=frequency)]).mean().reset_index()

    if d1.shape[0] != dataset.shape[0]:
        raise Exception("Error at site %d" % (dataset.loc[0, 'SiteId']))

    dataset = d1

    values = dataset['Value']

    outliers = dataset['Value'] > dataset['Value'].quantile(0.99)
    if np.sum(outliers)/values.shape[0] < 0.1:
        values = values.mask(outliers)

    null_val_indices,  = np.where(np.isnan(values))

    if frequency == 'D':
        offset = 7
        per_day_stride = 1
    elif frequency == '900s':
        offset = 7*24*4
        per_day_stride = 4*24
    elif frequency == 'h':
        offset = 7*24
        per_day_stride = 24
    else:
        raise Exception("Unknown frequency %s" % (frequency, ))

    for i in null_val_indices:
        num_indices = 30
        indices = [i - k * offset for k in range(1, num_indices+1)] + [i + k * offset for k in range(1, num_indices+1)]
        forecast_range,  = np.where(dataset['ForecastId'] == dataset.loc[i, 'ForecastId'])

        values.iloc[i] = np.nanmean(values.iloc[list(filter(lambda v: forecast_range[0] <= v < forecast_range[-1], indices))])

    if 0 < np.sum(np.isnan(values)) <= max(12*4, values.shape[0]*0.05):
        values = values.interpolate('linear')

    if np.sum(np.isnan(values)) > 0:
        logging.warning("Cannot fill values at locations %s" % (dataset.iloc[np.where(np.isnan(values))[0], :], ))

    dataset['Consumption'] = values

    consumption_agg_columns = []
    if frequency == 'D' or frequency == 'h' or frequency == '900s':
        dataset['ConsumptionDailyMean'] = dataset.groupby('ForecastId')['Consumption'] \
            .rolling(per_day_stride, min_periods=None).mean().shift(1).values
        dataset['ConsumptionWeeklyMean'] = dataset.groupby('ForecastId')['Consumption'] \
            .rolling(7 * per_day_stride, min_periods=None).mean().shift(1).values

        consumption_agg_columns += ['ConsumptionDailyMean', 'ConsumptionWeeklyMean']
    if frequency == 'D' or frequency == 'h':
        dataset['ConsumptionBiWeeklyMean'] = dataset.groupby('ForecastId')['Consumption'] \
            .rolling(7 * 2 * per_day_stride, min_periods=None).mean().shift(1).values
        dataset['ConsumptionMonthlyMean'] = dataset.groupby('ForecastId')['Consumption'] \
            .rolling(30 * per_day_stride, min_periods=None).mean().shift(1).values

        consumption_agg_columns += ['ConsumptionBiWeeklyMean', 'ConsumptionMonthlyMean']
    if frequency == 'h' or frequency == '900s':
        dataset['ConsumptionHalfDayMean'] = dataset.groupby('ForecastId')['Consumption'] \
            .rolling(per_day_stride // 2, min_periods=None).mean().shift(1).values
        dataset['ConsumptionQuarterDayMean'] = dataset.groupby('ForecastId')['Consumption'] \
            .rolling(per_day_stride // 4, min_periods=None).mean().shift(1).values

        consumption_agg_columns += ['ConsumptionHalfDayMean', 'ConsumptionQuarterDayMean']
    if frequency == '900s':
        dataset['ConsumptionHourlyMean'] = dataset.groupby('ForecastId')['Consumption'] \
            .rolling(per_day_stride // 24, min_periods=None).mean().shift(1).values

        consumption_agg_columns.append('ConsumptionHourlyMean')

    for column in consumption_agg_columns:
        dataset[column] = dataset.groupby('ForecastId')[column]\
            .transform(lambda x: x.fillna(x[x.first_valid_index()])).values

        dataset["%sPerSurfaceArea" % (column, )] = dataset[column] / dataset['SurfaceArea']
        dataset["%sPerTemperatureDiff" % (column,)] = \
            dataset[column] / np.maximum((dataset['TemperatureMean'] - dataset['BaseTemperature'])**2, 1e-3)

    return dataset


def build_test_y_dep_features(dataset, frequency, train_dataset, **kwargs):
    from src.utils.constants import get_output_window_size

    output_window_size = get_output_window_size(frequency)

    train_dataset = train_dataset[:]
    train_dataset['ForecastId'] = train_dataset['ForecastId'] + dataset['ForecastId'].max()

    merged_data = train_dataset.append(dataset)
    merged_data = merged_data.sort_values('Timestamp')

    if frequency == 'D':
        freq = np.timedelta64(1, 'D')
    elif frequency == 'h':
        freq = np.timedelta64(1, 'h')
    elif frequency == '900s':
        freq = np.timedelta64(900, 's')
    else:
        raise Exception('Unknown frequency %s' % (frequency,))

    if np.max(merged_data['Timestamp'] - merged_data['Timestamp'].shift(1)) >= 2 * freq:
        raise Exception("There are voids of upto %s after merging train and test data for site %s" %
                        (np.max(merged_data['Timestamp'] - merged_data['Timestamp'].shift(1)), train_dataset['SiteId'].iloc[0]))

    y_dep_features = []
    if frequency == 'D' or frequency == 'h' or frequency == '900s':
        y_dep_features += ['ConsumptionDailyMean', 'ConsumptionWeeklyMean']
    if frequency == 'D' or frequency == 'h':
        y_dep_features += ['ConsumptionBiWeeklyMean', 'ConsumptionMonthlyMean']
    if frequency == 'h' or frequency == '900s':
        y_dep_features += ['ConsumptionHalfDayMean', 'ConsumptionQuarterDayMean']
    if frequency == '900s':
        y_dep_features += ['ConsumptionHourlyMean']

    y_dep_sa_features = list(map(lambda v: "%sPerSurfaceArea" % (v, ), y_dep_features))
    y_dep_temp_features = list(map(lambda v: "%sPerTemperatureDiff" % (v,), y_dep_features))

    y_dep_features += (y_dep_sa_features + y_dep_temp_features)

    merged_data[y_dep_features] = merged_data[y_dep_features].shift(output_window_size + 1)

    new_test_data = merged_data.set_index('obs_id').loc[dataset['obs_id'], :].reset_index()

    if new_test_data.shape[0] != dataset.shape[0]:
        raise Exception("Missing data in new test data")

    new_test_data = new_test_data.drop(
        columns=list(set(train_dataset.keys()) - set(dataset.keys()) - set(y_dep_features)))

    return new_test_data


def build_features(builders, frequency, dataset, weather, metadata, holidays, train_dataset):
    from src.utils.data import ensure_no_na

    logger = logging.getLogger(__name__)

    sites = dataset.groupby('SiteId')['SiteId'].first().values

    result = []
    src_keys = dataset.keys()

    for site in tqdm(sites):
        site_data = dataset.loc[dataset['SiteId'] == site, :]
        site_weather_data = weather.loc[weather['SiteId'] == site, :]
        site_holiday_data = holidays.loc[holidays['SiteId'] == site, :]
        site_metadata = metadata.loc[metadata['SiteId'] == site, :].iloc[0]
        site_train_data = train_dataset.loc[train_dataset['SiteId'] == site, :] if train_dataset is not None else None

        res = reduce(lambda acc, v: v(dataset=acc, weather=site_weather_data, holidays=site_holiday_data,
                                     metadata=site_metadata, frequency=frequency, train_dataset=site_train_data),
                     builders, site_data)

        try:
            ensure_no_na(res.drop(columns=src_keys))
        except Exception as e:
            logger.error("Columns with empty values when processing site %d" % (site, ))
            raise Exception(e)

        result.append(res)

    return pd.concat(result, axis=0, ignore_index=True)


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('weather_filepath', type=click.Path(exists=True))
@click.argument('metadata_filepath', type=click.Path(exists=True))
@click.argument('holidays_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.option('--is_test_data', type=click.BOOL, default=False)
@click.option('--train_data_filepath', type=click.Path(), default=None)
@click.option('--frequency', type=click.STRING, default='D')
@click.option('--sites', type=click.STRING, default=None)
@click.option('--resume_site', type=click.INT, default=None)
def main(input_filepath, weather_filepath, metadata_filepath, holidays_filepath, output_filepath, frequency,
         is_test_data, train_data_filepath, sites, resume_site):
    """ Derives features for specified frequency
    """
    logger = logging.getLogger(__name__)

    logger.info('Reading %s' % (input_filepath, ))
    dataset = pd.read_hdf(input_filepath, "data", parse_dates=[2])

    logger.info('Reading %s' % (weather_filepath,))
    weather = pd.read_hdf(weather_filepath, "data", parse_dates=[1])

    logger.info('Reading %s' % (metadata_filepath,))
    metadata = pd.read_csv(metadata_filepath)

    logger.info('Reading %s' % (holidays_filepath,))
    holidays = pd.read_hdf(holidays_filepath, 'data', parse_dates=[0])

    if is_test_data:
        logger.info('Reading %s' % (train_data_filepath,))
        train_dataset = pd.read_hdf(train_data_filepath, "data", parse_dates=[1])
    else:
        train_dataset = None

    logger.info('Generating features')

    pd.options.mode.chained_assignment = None

    builders = [
        build_timestamp,
        build_holidays,
        build_temperature,
        build_site_metadata,
        build_consumption_value if not is_test_data else build_test_y_dep_features,
    ]

    if sites:
        sites = list(map(int, sites.split(',')))
        dataset = dataset.loc[np.isin(dataset['SiteId'], sites), :]

    if resume_site:
        dataset = dataset.loc[dataset['SiteId'] >= resume_site, :]

    features = build_features(builders, frequency, dataset, weather, metadata, holidays, train_dataset)

    logger.info("Saving to %s" % (output_filepath, ))
    features.to_hdf(output_filepath, "data", index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    sys.path.insert(0, project_dir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()