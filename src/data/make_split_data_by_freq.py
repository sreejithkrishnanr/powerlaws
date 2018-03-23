# -*- coding: utf-8 -*-
import os
import sys
import click
import logging
from dotenv import find_dotenv, load_dotenv

import pandas as pd
import numpy as np

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('freq_dataset_filepath', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--output_file_prefix', type=click.STRING, default=None)
def main(input_filepath, freq_dataset_filepath, output_dir, output_file_prefix):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    from src.utils.data import ensure_no_na

    logger = logging.getLogger(__name__)

    logger.info('Reading %s' % (input_filepath,))
    dataset = pd.read_csv(input_filepath, parse_dates=[2])

    logger.info('Reading %s' % (freq_dataset_filepath,))
    freq_data = pd.read_csv(freq_dataset_filepath)

    frequencies = freq_data.groupby('ForecastPeriodNS')['ForecastPeriodNS'].first().values

    logger.info('Applying frequency to dataset')

    for frequency in frequencies:
        forecast_ids = freq_data.loc[freq_data['ForecastPeriodNS'] == frequency, 'ForecastId'].values
        dataset.loc[np.isin(dataset['ForecastId'].values, forecast_ids), 'Frequency'] = frequency

    ensure_no_na(dataset['Frequency'])

    logger.info("Saving split dataset")
    input_file_name = os.path.splitext(os.path.basename(input_filepath))[0]
    output_file_prefix = output_file_prefix if output_file_prefix else input_file_name
    for frequency in frequencies:
        output_path = os.path.join(output_dir, "%s_%d.hd5" % (output_file_prefix, frequency))
        logger.info("Saving dataset for frequency %d to %s" % (frequency, output_path))

        dataset.loc[dataset['Frequency'] == frequency, :].to_hdf(output_path, "data", index=False)

    logger.info("Saved files to %s" % (output_dir,))


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
