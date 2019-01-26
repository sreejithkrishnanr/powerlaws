# -*- coding: utf-8 -*-
import os
import sys
import click
import logging
from dotenv import find_dotenv, load_dotenv
from collections import OrderedDict

import pandas as pd


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Process weather.csv
    """
    from src.utils.data import ensure_no_na

    logger = logging.getLogger(__name__)
    logger.info('Reading %s' % (input_filepath,))
    dataset = pd.read_csv(input_filepath, parse_dates=[1])

    logger.info("Processing weather data")
    dataset['Weight'] = 1 / dataset['Distance']
    dataset['Weighted_Temperature'] = dataset['Temperature'] * dataset['Weight']

    groups = dataset.groupby(['SiteId', 'Timestamp'])

    dataset = pd.DataFrame(data=OrderedDict([
        ('DistanceMean', groups['Distance'].mean()),
        ('DistanceMin', groups['Distance'].min()),
        ('DistanceMax', groups['Distance'].max()),
        ('DistanceVariance', groups['Distance'].std().fillna(0)),
        ('NumStations', groups['Distance'].count()),
        ('TemperatureMean', groups['Weighted_Temperature'].sum() / groups['Weight'].sum()),
        ('TemperatureVariance', groups['Temperature'].std().fillna(0)),
        ('TemperatureMin', groups['Temperature'].min()),
        ('TemperatureMax', groups['Temperature'].max()),
    ]))
    dataset = dataset.reset_index()

    ensure_no_na(dataset)

    logger.info("Saving to %s" % (output_filepath,))
    dataset.to_hdf(output_filepath, "data", index=False)


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
