from sklearn.cluster import KMeans
from sklearn.linear_model import SGDClassifier
from collections import Counter
from scipy.stats.mstats import mode

def PerformKMeansAnalysis(flattened_train_data, train_labels):
    naive_grouping = KMeans(n_clusters=10, n_init=5, n_jobs=8)
    naive_grouping.fit(flattened_train_data)
    group_labels = naive_grouping.labels_

    for label in set(group_labels):
        train_mask = group_labels == label
        real_labels = train_labels[train_mask]
        pos_stat = Counter(real_labels).values()
        pos_stat = np.array(pos_stat).astype(np.float)
        pos_stat /= real_labels.shape[0]

        representative = mode(real_labels)[0]
        gini_index = np.dot(pos_stat.T, np.ones(pos_stat.shape[0]) - pos_stat)
        print "For label %d: %s" % (label, gini_index)
        print "Representative: ", representative
        print "======"

def ApplySGDClassifier(flattened_train_data, train_labels, flattened_test_data, test_labels):
    svc = SGDClassifier(loss='squared_hinge', verbose=1, n_jobs=8, n_iter=100)
    svc.fit(flattened_train_data, train_labels)
    predictions = svc.predict(flattened_test_data)
    wrong = np.nonzero(test_labels - predictions)[0].astype(np.float).shape[0]
    print (float(wrong) / predictions.shape[0])

def NaiveMethodsOverview():
    train, validation, test = data()

    flattened_train_data = np.reshape(train.data_sources[0], (40000, 3072))
    train_labels = train.data_sources[1].T.ravel()

    flattened_validation_data = np.reshape(validation.data_sources[0], (10000, 3072))
    validation_labels = validation.data_sources[1].T.ravel()

    flattened_test_data = np.reshape(test.data_sources[0], (10000, 3072))
    test_labels = test.data_sources[1].T.ravel()

    PerformKMeansAnalysis(flattened_train_data, train_labels)
    ApplySGDClassifier(flattened_train_data, train_labels,
                       flattened_validation_data, validation_labels)

