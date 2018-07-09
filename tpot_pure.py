import numpy as np
import pandas as pd
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

# Score on the training set was:-0.006897601822671224
exported_pipeline = make_pipeline(
    PCA(iterated_power=2, svd_solver="randomized"),
    FeatureAgglomeration(affinity="euclidean", linkage="ward"),
    ElasticNetCV(l1_ratio=1.0, tol=0.01)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
