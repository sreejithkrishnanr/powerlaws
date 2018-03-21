# -*- coding: utf-8 -*-
import os
import sys
import click
import logging
import math
import matplotlib.pyplot as plt

from dotenv import find_dotenv, load_dotenv

import pandas as pd
import numpy as np

from tqdm import tqdm

from src.visualization.visualize import visualize_model_prediction
from src.utils.data import ensure_no_na


def generate_test_train_split(x, y, groups, obs_ids, output_window_size):
    min_test_size = output_window_size * 2.5
    fids, counts = np.unique(groups, return_counts=True)

    testable_fids = fids[np.where(counts >= min_test_size)]
    test_mask = np.repeat([False], groups.shape[0])

    test_size = output_window_size
    for fid in testable_fids:
        indices, = np.where(groups == fid)
        test_mask[indices[-test_size:]] = True

    train_mask = np.logical_not(test_mask)

    return x[train_mask], y[train_mask], groups[train_mask], obs_ids[train_mask], \
           x[test_mask], y[test_mask], groups[test_mask], obs_ids[test_mask]


def metrics_nwrmse(y_truth, y_pred, forecast_ids):
    errors = []

    fids = np.unique(forecast_ids)
    for fid in fids:
        forecast_y_pred = y_pred[forecast_ids == fid]
        forecast_y_truth = y_truth[forecast_ids == fid]

        count = 200
        weights = (3 * count - 2 * np.arange(1, count + 1) + 1) / (2 * count ** 2)

        weights = weights[:forecast_y_pred.shape[0]]

        error = np.sqrt(np.sum(((forecast_y_truth - forecast_y_pred) ** 2) * weights))
        mean_error = error / np.average(forecast_y_truth)
        errors.append(mean_error)

    return np.average(errors)


def prepare_test_data(test_data, train_data, frequency, output_window_size):
    sites = test_data.groupby('SiteId')['SiteId'].first().values

    result = pd.DataFrame()

    for site_id in tqdm(sites):
        site_test_data = test_data.loc[test_data['SiteId'] == site_id, :]

        site_train_data = train_data.loc[train_data['SiteId'] == site_id, :]
        site_train_data['ForecastId'] = site_train_data['ForecastId'] + site_test_data['ForecastId'].max()

        site_data = site_train_data.append(site_test_data)
        site_data = site_data.sort_values('Timestamp')

        if frequency == 'D':
            freq = np.timedelta64(1, 'D')
        elif frequency == 'h':
            freq = np.timedelta64(1, 'h')
        elif frequency == '900s':
            freq = np.timedelta64(900, 's')
        else:
            raise Exception('Unknown frequency %s' % (frequency, ))

        if np.max(site_data['Timestamp'] - site_data['Timestamp'].shift(1)) >= max(2*freq, np.timedelta64(2, 'h')):
            raise Exception("There are voids of upto %s after merging train and test data for site %s" %
                            (np.max(site_data['Timestamp'] - site_data['Timestamp'].shift(1)), site_id))

        y_dep_features = [
            'ConsumptionDailyMean',
            'ConsumptionWeeklyMean',
            'ConsumptionBiWeeklyMean',
            'ConsumptionMonthlyMean',
            'ConsumptionDailyMeanPerSurfaceArea',
            'ConsumptionWeeklyMeanPerSurfaceArea',
            'ConsumptionBiWeeklyMeanPerSurfaceArea',
            'ConsumptionMonthlyMeanPerSurfaceArea',
            'ConsumptionDailyMeanPerTemperatureDiff',
            'ConsumptionWeeklyMeanPerTemperatureDiff',
            'ConsumptionBiWeeklyMeanPerTemperatureDiff',
            'ConsumptionMonthlyMeanPerTemperatureDiff',
        ]

        site_data[y_dep_features] = site_data[y_dep_features].shift(output_window_size + 1)

        new_test_data = site_data.set_index('obs_id').loc[site_test_data['obs_id'], :].reset_index()

        if new_test_data.shape[0] != site_test_data.shape[0]:
            raise Exception("Missing data in new test data")

        new_test_data = new_test_data.drop(
            columns=list(set(train_data.keys()) - set(test_data.keys()) - set(y_dep_features)))
        ensure_no_na(new_test_data[y_dep_features])

        result = result.append(new_test_data, ignore_index=True)

    return result


train_data = pd.read_csv('data/processed/train_900000000000.csv', parse_dates=[1])
test_data = pd.read_csv('data/processed/test_900000000000.csv', parse_dates=[2])
result = prepare_test_data(test_data, train_data, '900s', 192)


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    dataset = pd.read_csv(input_filepath, parse_dates=[13])


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
