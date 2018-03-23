# -*- coding: utf-8 -*-
import os
import sys
import click
import logging
from dotenv import find_dotenv, load_dotenv

import pandas as pd
import numpy as np


@click.command()
@click.argument('submission_format_filepath', type=click.Path(exists=True))
@click.argument('input_freq1D_filepath', type=click.Path(exists=True))
@click.argument('input_freq1h_filepath', type=click.Path(exists=True))
@click.argument('input_freq900s_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(submission_format_filepath, input_freq1d_filepath, input_freq1h_filepath,
         input_freq900s_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)

    logger.info('Reading %s' % (submission_format_filepath, ))
    submission_format = pd.read_csv(submission_format_filepath)

    logger.info('Reading %s' % (input_freq1d_filepath,))
    freq1D = pd.read_hdf(input_freq1d_filepath, "data")

    logger.info('Reading %s' % (input_freq1h_filepath,))
    freq1h = pd.read_hdf(input_freq1h_filepath, "data")

    logger.info('Reading %s' % (input_freq900s_filepath,))
    freq900s = pd.read_hdf(input_freq900s_filepath, "data")

    logger.info('Merging predictions')
    predictions = freq1D.append(freq1h).append(freq900s).set_index('obs_id')['Value'].loc[submission_format['obs_id']]

    assert predictions.shape[0] == submission_format.shape[0]

    submission_format['Value'] = predictions.values

    logger.info('Saving to %s' % (output_filepath,))
    submission_format.to_csv(output_filepath, index=False)


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
