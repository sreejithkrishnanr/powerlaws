import logging


def ensure_no_na(data):
    logger = logging.getLogger(__name__)

    if data.isnull().values.any():
        for column in data.keys():
            if data[column].isnull().values.any():
                logger.error("Column %s has %d null values" % (column, data[column].isnull().values.sum()))

        raise Exception("Dataset have null values")
