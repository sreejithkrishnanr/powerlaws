# -*- coding: utf-8 -*-
import os
import sys
import click
import logging
import json

from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm

from tabulate import tabulate
import pandas as pd
import numpy as np


def generate_test_train_split(x, y, groups, output_window_size):
    min_test_size = output_window_size * 2.5
    fids, counts = np.unique(groups, return_counts=True)

    testable_fids = fids[np.where(counts >= min_test_size)]
    test_mask = np.repeat([False], groups.shape[0])

    test_size = output_window_size
    for fid in testable_fids:
        indices, = np.where(groups == fid)
        test_mask[indices[-test_size:]] = True

    train_mask = np.logical_not(test_mask)

    return x[train_mask], y[train_mask], groups[train_mask], x[test_mask], y[test_mask], groups[test_mask]


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


def build_model_per_site(train_data, frequency, output_dir, evaluate_only=False, models=('gb',), sites=None):
    logger = logging.getLogger(__name__)

    sites = train_data.groupby('SiteId')['SiteId'].first().values if sites is None else sites
    output_window_size = get_output_window_size(frequency)

    scores = {}
    for site in tqdm(sites):
        site_train_data = train_data.loc[train_data['SiteId'] == site, :]
        x, y, groups = select_features(site_train_data, frequency, novalue=False, include_site_id=False,
                                       use_consumption_per_sa=False)
        x_train, y_train, g_train, x_test, y_test, g_test = generate_test_train_split(x, y, groups, output_window_size)

        for model in models:
            logger.info("Training model %s for site %s" % (model, site))

            evaluate, build, predict = MODEL_REGISTRY[model]

            y_pred = evaluate(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                              site_id=site, frequency=frequency)

            score = metrics_nwrmse(y_test, y_pred, g_test)

            if model not in scores.keys():
                scores[model] = []

            scores[model].append(score)

            logger.info("Model %s scored %f for site %s" % (model, score, site))

            if evaluate_only:
                continue

            model_key = "model_%s" % (model,)
            if model_key not in scores.keys():
                score[model_key] = []

            output_path = os.path.abspath(
                os.path.join(output_dir, "model_%s_%s_%s_%f.pkl" % (model, frequency, site, round(score, 4))))

            scores[model_key].append(output_path)

            logger.info("Building and saving model to %s" % (output_path,))
            build(x=x, y=y, site_id=site, frequency=frequency, output_path=output_path)

    scores['sites'] = sites.tolist()

    schema = scores
    scores = pd.DataFrame(data=scores)[['sites'] + models]

    logger.info(tabulate(scores, headers='keys', tablefmt='psql'))

    report_path = os.path.join(output_dir, 'scores.html')
    logger.info('Scores saved to %s' % (report_path,))
    scores.to_html(report_path)

    logger.info('Mean: %s' % (scores[models].mean(), ))

    if not evaluate_only:
        schema_path = os.path.join(output_dir, 'schema.json')
        logger.info("Schema saved to %s" % (schema_path,))
        with open(schema_path, 'w') as f:
            json.dump(schema, f)


@click.command()
@click.argument('train_data_filepath', type=click.Path(exists=True))
@click.argument('output_folder', type=click.Path())
@click.option('--frequency', type=click.STRING, default='D')
@click.option('--sites', type=click.STRING, default=None)
@click.option('--models', type=click.STRING, default="gb")
@click.option('--evaluate_only', type=click.BOOL, default=False)
def main(train_data_filepath, output_folder, frequency, sites, models, evaluate_only):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)

    logger.info("Reading %s" % (train_data_filepath,))
    train_data = pd.read_csv(train_data_filepath, parse_dates=[1])
    sites = np.array(list(map(int, sites.split(',')))) if sites is not None else None
    models = models.split(',') if models is not None else None

    os.makedirs(output_folder, exist_ok=True)

    build_model_per_site(train_data, frequency, output_folder, evaluate_only=evaluate_only, models=models, sites=sites)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    sys.path.insert(0, project_dir)

    from src.models.model_common import select_features, get_output_window_size, MODEL_REGISTRY

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
