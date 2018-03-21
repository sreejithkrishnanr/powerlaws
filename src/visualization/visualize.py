import matplotlib.pyplot as plt


def visualize_model_prediction(dataset, y_train_pred, id_train, y_test_pred, id_test):
    site = dataset['SiteId'].iloc[0]

    dataset = dataset.set_index('obs_id')
    train_timestamps = dataset.loc[id_train, 'Timestamp']
    test_timestamps = dataset.loc[id_test, 'Timestamp']

    plt.plot(dataset['Timestamp'], dataset['Consumption'], color='green', label='True')
    plt.plot(train_timestamps, y_train_pred, color='blue', label='Train')
    plt.plot(test_timestamps, y_test_pred, color='red', label='Test')
    plt.title("Site %s" % (site, ))
    plt.show()
