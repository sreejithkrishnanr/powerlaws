# -*- coding: utf-8 -*-
import os
import sys
import click
import logging
from dotenv import find_dotenv, load_dotenv

import pandas as pd
import numpy as np

from tqdm import tqdm


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True)) # holidays.csv
@click.argument('metadata_filepath', type=click.Path(exists=True)) # metadata.csv
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath, metadata_filepath):
    """ Process holidays.csv
    """
    logger = logging.getLogger(__name__)

    logger.info('Reading %s' % (input_filepath, ))
    holidays = pd.read_csv(input_filepath, parse_dates=[1])

    logger.info('Reading %s' % (metadata_filepath,))
    metadata = pd.read_csv(metadata_filepath)

    sites = metadata.groupby('SiteId')['SiteId'].first().values

    rolls = [2, 3, 4, 5, 6, 7, 10, 15, 30, 120]

    start_date = np.datetime64('2009-01-01')
    end_date = np.datetime64('2018-01-30')
    timesteps = np.arange(start_date, end_date, np.timedelta64(1, 'D'))
    time_data = pd.DataFrame()
    time_data['Date'] = timesteps
    time_data['DayOfWeek'] = time_data['Date'].dt.dayofweek
    time_data['WeekOfYear'] = time_data['Date'].dt.weekofyear
    time_data['Year'] = time_data['Date'].dt.year
    time_data['Month'] = time_data['Date'].dt.month
    time_data['Quarter'] = time_data['Date'].dt.quarter

    result = []
    for site in tqdm(sites):
        dataset = time_data.copy()

        site_holidays = holidays.loc[holidays['SiteId'] == site, :]
        site_metadata = metadata.loc[metadata['SiteId'] == site, :].iloc[0]
        site_weekends, = np.where(site_metadata['MondayIsDayOff':'SundayIsDayOff'])

        sp_holiday_key = "SpecialHoliday"
        weekend_key = "Weekend"
        holiday_key = "Holiday"

        dataset[sp_holiday_key] = False
        dataset[sp_holiday_key].loc[np.isin(dataset['Date'].values, site_holidays['Date'].values)] = True

        dataset[weekend_key] = False
        dataset[weekend_key] = np.isin(dataset['DayOfWeek'], site_weekends)

        dataset[holiday_key] = dataset[sp_holiday_key].values | dataset[weekend_key].values

        for key in [sp_holiday_key, weekend_key, holiday_key]:
            dataset["%sYearCount" % (key, )] = dataset.groupby('Year')[key].transform(
                lambda v: np.repeat(v.sum(), v.shape[0])).values.astype(int)

            dataset["%sRemainingYearCount" % (key,)] = \
                dataset["%sYearCount" % (key, )].values - dataset.groupby('Year')[key].cumsum().values

            dataset["%sQuarterCount" % (key, )] = dataset.groupby(['Year', 'Quarter'])[key].transform(
                lambda v: np.repeat(v.sum(), v.shape[0])).values.astype(int)
            dataset["%sRemainingQuarterCount" % (key,)] = \
                dataset["%sQuarterCount" % (key,)].values - dataset.groupby(['Year', 'Quarter'])[key].cumsum().values

            dataset["%sMonthCount" % (key,)] = dataset.groupby(['Year', 'Month'])[key].transform(
                lambda v: np.repeat(v.sum(), v.shape[0])).values.astype(int)
            dataset["%sRemainingMonthCount" % (key,)] = \
                dataset["%sMonthCount" % (key,)].values - dataset.groupby(['Year', 'Month'])[key].cumsum().values

        for roll in rolls:
            for key in [sp_holiday_key, weekend_key, holiday_key]:
                dataset['Num%sWithin%d' % (key, roll,)] = \
                    dataset[key].rolling(roll, center=True, min_periods=1).sum()

        dataset['SiteId'] = site

        result.append(dataset)

    result = pd.concat(result, axis=0, ignore_index=True)

    logger.info("Saving to %s" % (output_filepath, ))
    result.to_hdf(output_filepath, 'data')


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
