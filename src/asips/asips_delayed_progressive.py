from skmultiflow.data import DataStream
from sklearn.metrics import accuracy_score
from readerwriterlock import rwlock
from skmultiflow.data import DataStream
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from readerwriterlock import rwlock
from skmultiflow.evaluation import EvaluatePrequential
import numpy as np
import pandas as pd
import bisect
import multiprocessing as mp
from skmultiflow.trees.extremely_fast_decision_tree import ExtremelyFastDecisionTreeClassifier
import datetime
import multiprocessing as mp
from firebase import firebase
import asips_utils
from firebase import jsonutil
import json
from firebase_helper import FirebaseHelper

class Asips:
    """ Asips.
            efdtgh: ExtremelyFastDecisionTreeClassifier(split_criterion='gaussian_hellinger')
                Extremely Fast Decision Tree with the split criterion as gaussian_hellinger
            evaluator: EvaluatePrequential
                Evaluator for progressive predictions from GH-EFDT

    """
    def __init__(self):
        self.efdtgh = ExtremelyFastDecisionTreeClassifier(split_criterion='gaussian_hellinger')
        self.evaluator = EvaluatePrequential(show_plot=False,
                                             pretrain_size=0,
                                             max_samples=210,
                                             metrics=['kappa', 'gmean', 'accuracy', 'recall', 'precision', 'f1',
                                                 'running_time'],
                                             n_wait=20,
                                             output_file="../output_file")
        self.last_sample_count = 0
        # read csv
        print('-------READ CSV-------')
        data_frame = pd.read_csv('pulsar_data_train.csv')
        # partition stream for pretrain and 3*(predictions & train)
        pretrain_set = data_frame.iloc[:300, :]
        first_batch = data_frame.iloc[301:401]
        second_batch = data_frame.iloc[402:502, :]
        print("Shape of new partitioned dataframes - {} , {},{}".format(pretrain_set.shape, first_batch.shape,
                                                                        second_batch.shape))
        print(first_batch.head())
        # make streams
        print('-------INIT STREAMS-------')
        self.stream_pre = DataStream(allow_nan=True, data= pretrain_set)
        self.stream_first = DataStream(allow_nan=True, data= first_batch)
        self.stream_second = DataStream(allow_nan=True, data= second_batch)


    def evaluation_measures_skmultiflow(self):
        dict = self.evaluator.evaluation_summary() # current mean values
        current_measures = self.evaluator.get_current_measurements(model_idx=0) # current values

        FirebaseHelper.save_current_mean_measures(dict) # Save current mean measures
        FirebaseHelper.save_current_evaluation_measures(current_measures) # Save current evaluation measures
        FirebaseHelper.save_kappa_values(current_measures, dict) #keep kappa and mean kappa for the chart

        string = 'Kappa: {0} \nG-Mean: {0} \nAccuracy: {1} \nRecall: {2} \nPrecision: {3} \nF1 Score: {4}'
        log = string.format(current_measures.kappa_score(),current_measures.geometric_mean_score(), current_measures.accuracy_score(),
                  current_measures.recall_score(), current_measures.precision_score(), current_measures.f1_score())
        print(log)
        return log


    @staticmethod
    def print_stream_info(stream):
        print("Features:")
        print(stream.feature_names)
        print("Number of Targets: " + str(stream.n_targets) + "\nName: " + str(stream.target_names))
        print("Target class values: " + str(stream.target_values))


    def pretrain(self,count):
        print('-------PRETRAIN STARTED-------')
        stream = self.make_stream_by_count(count)
        Asips.print_stream_info(stream)
        n_samples = 0
        # Train the estimator with the samples provided by the data stream
        start_time = datetime.datetime.now()
        while stream.has_more_samples():
            X, y = stream.next_sample()
            self.efdtgh.predict(X)
            self.efdtgh.partial_fit(X, y)
            n_samples += 1
        end_time = datetime.datetime.now()
        time_difference = end_time - start_time
        print(type(time_difference))
        print('-------PRETRAIN FINISHED-------')
        string = 'Time difference: {0} \ninfo: {1} \n Number of learning samples: {2} \n Features: {3} \n' \
        'Target values: {4}'
        log = string.format(time_difference, str(self.efdtgh.get_info),
                            n_samples, stream.feature_names, str(stream.target_values))

        FirebaseHelper.save_pretrain_info(time_difference, self.efdtgh, n_samples, stream)

        print(log)
        return log

    def make_prediction(self, count, is_first_run = True):
        print(mp.current_process())
        stream = self.make_stream_by_count(count)
        self.evaluator.evaluate(stream=stream, model=self.efdtgh, is_first_run =is_first_run)
        print("info: ")
        print(self.efdtgh.get_info)
        return self.evaluation_measures_skmultiflow()

    def learn_from_all(self):
        # Read from firebase
        result = FirebaseHelper.get_signal_from_firebase('')
        s = json.dumps(result, cls=jsonutil.JSONEncoder)
        j = json.loads(s)

        # convert to a dataframe
        dataframe = pd.DataFrame.from_dict(j, orient='index')
        dataframe = Asips.preprocess(dataframe)

        # pass dataframe
        log = self.make_stream_and_learn(dataframe)
        #delete from predictions collection in the firebase
        FirebaseHelper.delete_signal_from_firebase('')
        return log

    def learn(self, id):
        #get data from firebase
        result = FirebaseHelper.get_signal_from_firebase(id)
        s = json.dumps(result, cls=jsonutil.JSONEncoder)
        j = json.loads(s)
        print(j)

        #convert to a dataframe
        dataframe = pd.DataFrame.from_dict([j], orient='columns')
        dataframe = Asips.preprocess(dataframe)
        #pass dataframe
        log = self.make_stream_and_learn(dataframe)
        # #delete from predictions collection in the firebase
        FirebaseHelper.delete_signal_from_firebase(id)
        return log


    def make_stream_and_learn(self, dataframe):
        # Get stream
        stream = Asips.make_stream(dataframe)

        # Learn and update database
        start_time = datetime.datetime.now()
        while stream.has_more_samples():
            X, y = stream.next_sample()
            self.efdtgh.partial_fit(X, y)
            data = {'meanOfTheIntegratedProfile': X[0][0],
                    'standardDeviationOfTheIntegratedProfile': X[0][1],
                    'excessKurtosisOfTheIntegratedProfile': X[0][2],
                    'skewnessOfTheIntegratedProfile': X[0][3],
                    'meanOfTheDMSNRCurve': X[0][4],
                    'standardDeviationOfTheDMSNRCurve': X[0][5],
                    'excessKurtosisOfTheDMSNRCurve': X[0][6],
                    'skewnessOfTheDMSNRCurve': X[0][7],
                    'targetClass': int(y[0]),
                    }
            # add to the confirmed collection in the firebase
            FirebaseHelper.save_signal_to_firebase(asips_utils.CONFIRMED_PATH, data)
        end_time = datetime.datetime.now()
        time_difference = end_time - start_time

        string = 'Time difference: {0} \ninfo: {1} \nNumber of learning samples: {2}'
        log = string.format(time_difference, self.efdtgh.get_info,
                            len(dataframe.index))
        print(log)
        return log

    @staticmethod
    def preprocess(dataframe):
        print(dataframe)
        dataframe = dataframe.rename(columns={'meanOfTheIntegratedProfile': 'Mean of the integrated profile',
                                              'standardDeviationOfTheIntegratedProfile': 'Standard deviation of the integrated profile',
                                              'excessKurtosisOfTheIntegratedProfile': 'Excess kurtosis of the integrated profile',
                                              'skewnessOfTheIntegratedProfile': 'Skewness of the integrated profile',
                                              'meanOfTheDMSNRCurve': 'Mean of the DM-SNR curve',
                                              'standardDeviationOfTheDMSNRCurve': 'Standard deviation of the DM-SNR curve',
                                              'excessKurtosisOfTheDMSNRCurve': 'Excess kurtosis of the DM-SNR curve',
                                              'skewnessOfTheDMSNRCurve': 'Skewness of the DM-SNR curve',
                                              'targetClass': 'target_class',
                                              'yPred': 'y_pred'})
        print(dataframe)
        return dataframe

    @staticmethod
    def make_stream(dataframe):
        # Preprocess dataframe to create a stream
        dataframe = dataframe.drop(columns='y_pred')
        if 'smoteBagging' in dataframe.columns:
            dataframe = dataframe.drop(columns='smoteBagging')
        if 'underOverBagging' in dataframe.columns:
            dataframe = dataframe.drop(columns='underOverBagging')

        dataframe = dataframe.dropna()
        dataframe['target_class'] = dataframe['target_class'].astype(int)
        dataframe = dataframe[dataframe.columns.reindex(asips_utils.DATAFRAME_HEAD)[0]]

        print(dataframe.head())

        # Make stream
        stream = DataStream(allow_nan=True, data=dataframe)
        Asips.print_stream_info(stream)
        return stream

    def make_stream_by_count(self, count):
        data_frame = pd.read_csv('pulsar_data_train.csv')
        self.last_sample_count += 1
        data_frame = data_frame.iloc[self.last_sample_count : self.last_sample_count + count]
        self.last_sample_count += count

        return DataStream(allow_nan=True, data= data_frame)

