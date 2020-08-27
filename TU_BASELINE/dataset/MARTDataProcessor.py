import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np


class MARTDataProcessor:

    def __init__(self):
        return None

    def fill_pd_missing_values(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        imputer = SimpleImputer(strategy='median', add_indicator=True)
        imputed_dataframe = imputer.fit_transform(dataframe)
        return imputed_dataframe
 
    def __mean_embedded_features(self, embedded_features: np.array) -> np.array:
        mean_embedded_feat = np.mean(embedded_features, axis=0)
        return mean_embedded_feat

    def get_autographer_mean_embedded_features(self, embedded_features):
        mean_embedded_features = []
        for feats in embedded_features:
            mean_feat = self.__mean_embedded_features(feats) 
            mean_embedded_features.append(mean_feat)
        mean_embedded_features = np.array(mean_embedded_features)
        return mean_embedded_features
