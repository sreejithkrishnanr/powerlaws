# -*- coding: utf-8 -*-
import os
import sys
import click
import logging
import zipfile

from dotenv import find_dotenv, load_dotenv

import pandas as pd
import numpy as np


@click.command()
# Path to raw submission_format.csv
@click.argument('submission_format_filepath', type=click.Path(exists=True))
# Path to predicted hd5 file for 86400000000000ns frequency
@click.argument('input_freq1D_filepath', type=click.Path(exists=True))
# Path to predicted hd5 file for 3600000000000ns frequency
@click.argument('input_freq1h_filepath', type=click.Path(exists=True))
# Path to predicted hd5 file for 900000000000ns frequency
@click.argument('input_freq900s_filepath', type=click.Path(exists=True))
# Path to output file
@click.argument('output_filepath', type=click.Path())
def main(submission_format_filepath, input_freq1d_filepath, input_freq1h_filepath,
         input_freq900s_filepath, output_filepath):
    """ Generate submission file
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

    if np.sum(np.isnan(predictions)) > 0:
        raise Exception("Predictions contains nan values")

    assert predictions.shape[0] == submission_format.shape[0]

    submission_format['Value'] = predictions.values

    logger.info('Saving to %s' % (output_filepath,))
    submission_format.to_csv(output_filepath, index=False)

    compressed_submission_filepath = output_filepath + '.zip'
    logger.info('Saving compressed %s to %s' % (output_filepath, compressed_submission_filepath))
    with zipfile.ZipFile(compressed_submission_filepath, mode='w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(output_filepath, 'submission.csv')


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
