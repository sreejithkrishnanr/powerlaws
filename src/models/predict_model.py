# -*- coding: utf-8 -*-
import os
import sys
import click
import logging
import json

from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm

import pandas as pd
import numpy as np


def _parse_schema(schema):
    parsed = {}
    for i, site in enumerate(schema['sites']):
        parsed[site] = []

        for key in schema.keys():
            if key.startswith('model_'):
                model_name = key[6:]
                model_path = schema[key][i]
                model_score = schema[model_name][i]
                parsed[site].append({
                    'name': model_name,
                    'path': model_path,
                    'score': model_score
                })

    return parsed


def _select_model_with_best_score(schema, site):
    models = schema[site]

    best_model = None

    for model in models:
        if best_model is None or best_model['score'] < model['score']:
            best_model = model

    return best_model


def make_predictions_with_schema(test_data, train_data, frequency, schema, select_model=None, sites=None):
    logger = logging.getLogger(__name__)

    sites = test_data.groupby('SiteId')['SiteId'].first().values if sites is None else sites

    result = pd.DataFrame()

    for site in tqdm(sites):
        site_test_data = test_data.loc[test_data['SiteId'] == site, :]
        site_train_data = train_data.loc[train_data['SiteId'] == site, :]

        model = _select_model_with_best_score(schema, site) if select_model is None else select_model
        name, path, score = model['name'], model['path'], model['score']

        logger.info('Using model %s with score %f' % (name, score))

        evaluate, build, predict = MODEL_REGISTRY[name]

        x, y, groups = select_features(site_test_data, frequency, novalue=True, train_data=site_train_data,
                                       include_site_id=False, use_consumption_per_sa=False)

        y_pred = predict(x_test=x, g_test=groups, model_path=path, frequency=frequency, site_id=site,
                         train_data=site_train_data, test_data=site_test_data)
        predictions = site_test_data[['obs_id', 'SiteId', 'Timestamp', 'ForecastId']]
        predictions.insert(4, 'Value', y_pred)

        result = result.append(predictions)

    return result


@click.command()
@click.argument('test_data_filepath', type=click.Path(exists=True))
@click.argument('train_data_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.option('--frequency', type=click.STRING, default='D')
@click.option('--schema_filepath', type=click.Path(), default=None)
@click.option('--sites', type=click.STRING, default=None)
def main(test_data_filepath, train_data_filepath, output_filepath, frequency, schema_filepath, sites):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)

    if not schema_filepath:
        raise Exception('No schema provided')

    logger.info("Reading %s" % (test_data_filepath, ))
    test_data = pd.read_hdf(test_data_filepath, "data", parse_dates=[74])

    logger.info("Reading %s" % (train_data_filepath,))
    train_data = pd.read_hdf(train_data_filepath, "data", parse_dates=[1])

    with open(schema_filepath, 'r') as f:
        schema = _parse_schema(json.load(f))

    sites = np.array(list(map(int, sites.split(',')))) if sites is not None else None
    predictions = make_predictions_with_schema(test_data, train_data, frequency, schema, sites=sites)

    logger.info('Saving to %s' % (output_filepath, ))
    predictions.to_hdf(output_filepath, "data", index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    sys.path.insert(0, project_dir)

    from src.models.model_common import MODEL_REGISTRY, select_features

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
