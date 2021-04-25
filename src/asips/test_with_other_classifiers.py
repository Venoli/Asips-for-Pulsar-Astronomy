from skmultiflow.meta import OnlineSMOTEBaggingClassifier
from skmultiflow.meta import OnlineUnderOverBaggingClassifier
from skmultiflow.data import DataStream
import numpy as np
from skmultiflow.evaluation import EvaluatePrequential
import datetime
from firebase_helper import FirebaseHelper
import asips_utils
from asips_delayed_progressive import Asips
import pandas as pd
from firebase import firebase
import asips_utils
from firebase import jsonutil
import json
from firebase_helper import FirebaseHelper
class TestWithOtherClassifiers:
    def __init__(self):
        self.online_smote_bagging = OnlineSMOTEBaggingClassifier()
        self.online_under_over_bagging = OnlineUnderOverBaggingClassifier()

    @staticmethod
    def preprocess(json_data):
            dataframe = pd.DataFrame.from_dict(json_data, orient='index')
            dataframe = dataframe.T
            dataframe = dataframe.rename(columns={'meanOfTheIntegratedProfile': 'Mean of the integrated profile',
                                                  'standardDeviationOfTheIntegratedProfile': 'Standard deviation of the integrated profile',
                                                  'excessKurtosisOfTheIntegratedProfile': 'Excess kurtosis of the integrated profile',
                                                  'skewnessOfTheIntegratedProfile': 'Skewness of the integrated profile',
                                                  'meanOfTheDMSNRCurve': 'Mean of the DM-SNR curve',
                                                  'standardDeviationOfTheDMSNRCurve': 'Standard deviation of the DM-SNR curve',
                                                  'excessKurtosisOfTheDMSNRCurve': 'Excess kurtosis of the DM-SNR curve',
                                                  'skewnessOfTheDMSNRCurve': 'Skewness of the DM-SNR curve', 'targetClass': 'target_class',
                                                  'yPred': 'y_pred'})
            print(dataframe)
            return  Asips.make_stream(dataframe)

    def pretrain(self):
        data_frame = pd.read_csv('pulsar_data_train.csv')
        pretrain_set = data_frame.iloc[:300, :]
        print(pretrain_set.head())
        stream_pre = DataStream(allow_nan=True, data=pretrain_set)

        while stream_pre.has_more_samples():
            X, y = stream_pre.next_sample()
            self.online_smote_bagging.predict(X)
            self.online_smote_bagging.partial_fit(X, y, np.array([0, 1]))
            self.online_under_over_bagging.predict(X)
            self.online_under_over_bagging.partial_fit(X, y, np.array([0, 1]))

    def classification_pipeline(self, model):
        # Read GH-EFDT predicted signals from db, then send the  same features to another model,
        # get the output, then update db by appending new predictions under same id.
        self.pretrain()
        predictions = FirebaseHelper.get_signal_from_firebase('')
        s = json.dumps(predictions, cls=jsonutil.JSONEncoder)
        json_list = json.loads(s)
        print(s)
        print(json_list)

        for j in json_list:
            print(j)
            print(json_list[j])
            stream = self.preprocess(json_list[j])
            if model == 'smote_bagging':
                y_pred = self.online_smote_bagging_classifier(stream)
                print(y_pred[0])
                print(type(y_pred[0]))
                FirebaseHelper.update_signal_record(j, 'smoteBagging', int(y_pred[0]))
            else:
                y_pred = self.online_under_over_bagging_classifier(stream)
                FirebaseHelper.update_signal_record(j, 'underOverBagging', int(y_pred[0]))

        return 'Done'


    def online_smote_bagging_classifier(self, stream):
        X, y = stream.next_sample()
        y_pred = self.online_smote_bagging.predict(X)
        self.online_smote_bagging.partial_fit(X, y, np.array([0, 1]))
        return y_pred


    def online_under_over_bagging_classifier(self, stream):
        X, y = stream.next_sample()
        y_pred = self.online_under_over_bagging.predict(X)
        self.online_under_over_bagging.partial_fit(X, y, np.array([0, 1]))
        return y_pred



