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
from skmultiflow.utils import asips_utils
from firebase import jsonutil
import json
class Asips:
    def __init__(self):
        self.efdtgh = ExtremelyFastDecisionTreeClassifier(split_criterion='gaussian_hellinger')
        self.evaluator = EvaluatePrequential(show_plot=False,
                                             pretrain_size=0,
                                             max_samples=210,
                                             metrics=['kappa', 'gmean', 'accuracy', 'recall', 'precision', 'f1',
                                                 'running_time'],
                                             n_wait=20,
                                             output_file="../output_file")
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



    def evaluation_measures_sklearn(self, y_true_all, y_pred_all):
        tn, fp, fn, tp = confusion_matrix(y_true_all, y_pred_all).ravel()
        recall = recall_score(y_true_all, y_pred_all)
        precision = precision_score(y_true_all, y_pred_all)
        accuracy = accuracy_score(y_true_all, y_pred_all)
        f1 = f1_score(y_true_all, y_pred_all)
        specificity = tn / (tn + fp)
        G_mean = np.sqrt((precision * specificity))

        print('Accuracy: {0} \nRecall: {1} \nPrecision: {2} \nF1 Score: {3} \nSpecificity(TNR): {4} \nG-Mean: {5}'
              .format(accuracy, recall, precision, f1, specificity, G_mean))


        # self.firebase.put_async('asips-3efdd-default-rtdb/CurrentEvaluationMeasures/',"accuracy", accuracy)
        # self.firebase.put_async('asips-3efdd-default-rtdb/CurrentEvaluationMeasures/',"recall", recall)
        # self.firebase.put_async('asips-3efdd-default-rtdb/CurrentEvaluationMeasures/',"precision", precision)
        # self.firebase.put_async('asips-3efdd-default-rtdb/CurrentEvaluationMeasures/',"fScore", f1)
        # self.firebase.put_async('asips-3efdd-default-rtdb/CurrentEvaluationMeasures/',"specificity", specificity)
        # self.firebase.put_async('asips-3efdd-default-rtdb/CurrentEvaluationMeasures/',"gMean", G_mean)
        # self.firebase.put_async('asips-3efdd-default-rtdb/CurrentEvaluationMeasures/',"confirmationState", 0)

    def evaluation_measures_skmultiflow(self):
        current_measures = self.evaluator.get_current_measurements(model_idx=0)
        eval_measures_path = asips_utils.BASE_PATH + asips_utils.CURRENT_EVALUATION_MEASURES_PATH
        mean_performance_path = asips_utils.BASE_PATH + asips_utils.MEAN_PERFORMANCES_PATH

        asips_utils.FIREBASE_REF.put_async(eval_measures_path, "kappa", current_measures.kappa_score())
        asips_utils.FIREBASE_REF.put_async(eval_measures_path, "gMean", current_measures.geometric_mean_score())
        asips_utils.FIREBASE_REF.put_async(eval_measures_path, "accuracy", current_measures.accuracy_score())
        asips_utils.FIREBASE_REF.put_async(eval_measures_path, "recall", current_measures.recall_score())
        asips_utils.FIREBASE_REF.put_async(eval_measures_path, "precision", current_measures.precision_score())
        asips_utils.FIREBASE_REF.put_async(eval_measures_path, "fScore", current_measures.f1_score())
        asips_utils.FIREBASE_REF.put_async(eval_measures_path, "samples", current_measures.n_samples)
        asips_utils.FIREBASE_REF.put_async(eval_measures_path, "confirmationState", 0)
        result = asips_utils.FIREBASE_REF.get(mean_performance_path, 'kappa')
        print(len(result))
        asips_utils.FIREBASE_REF.put(mean_performance_path + 'kappa', len(result), current_measures.kappa_score())
        string = 'Kappa: {0} \nG-Mean: {0} \nAccuracy: {1} \nRecall: {2} \nPrecision: {3} \nF1 Score: {4}'
        log = string.format(current_measures.kappa_score(),current_measures.geometric_mean_score(), current_measures.accuracy_score(),
                  current_measures.recall_score(), current_measures.precision_score(), current_measures.f1_score())
        print(log)
        return log

    def print_stream_info(self, stream):
        print("Features:")
        print(stream.feature_names)
        print("Number of Targets: " + str(stream.n_targets) + "\nName: " + str(stream.target_names))
        print("Target class values: " + str(stream.target_values))

    def pretrain(self):
        print('-------PRETRAIN STARTED-------')
        self.print_stream_info(self.stream_pre)
        n_samples = 0
        # Train the estimator with the samples provided by the data stream
        start_time = datetime.datetime.now()
        while self.stream_pre.has_more_samples():
            X, y = self.stream_pre.next_sample()
            self.efdtgh.predict(X)
            self.efdtgh.partial_fit(X, y)
            n_samples += 1
        end_time = datetime.datetime.now()
        time_difference = end_time - start_time
        print('-------PRETRAIN FINISHED-------')
        string = 'Time difference: {0} \ninfo: {1} \n Number of learning samples: {2}'
        log = string.format(time_difference, str(self.efdtgh.get_info),
                            n_samples)
        print(log)
        return log

    def make_prediction(self, stream, is_first_run = True):
        print(mp.current_process())
        self.evaluator.evaluate(stream=stream, model=self.efdtgh, is_first_run =is_first_run)
        print("info: ")
        print(self.efdtgh.get_info)
        return self.evaluation_measures_skmultiflow()

    def learn_from_all(self):
        # Read from firebase
        result = asips_utils.FIREBASE_REF.get(asips_utils.BASE_PATH + asips_utils.PREDICTIONS_PATH, '')
        s = json.dumps(result, cls=jsonutil.JSONEncoder)
        j = json.loads(s)
        print(s)
        print(len(s))
        print(len(j))
        print(j)

        # convert to a dataframe
        dataframe = pd.DataFrame.from_dict(j, orient='index')
        dataframe.columns = asips_utils.DATAFRAME_HEAD
        print(dataframe)

        # pass dataframe
        log = self.make_stream_and_learn(dataframe)
        #delete from predictions collection in the firebase
        asips_utils.FIREBASE_REF.delete(asips_utils.BASE_PATH + asips_utils.PREDICTIONS_PATH, '')
        return log

    def learn(self, id):
        #get data from firebase
        result = asips_utils.FIREBASE_REF.get(asips_utils.BASE_PATH + asips_utils.PREDICTIONS_PATH, id)
        s = json.dumps(result, cls=jsonutil.JSONEncoder)
        j = json.loads(s)
        print(j)
        print(s)
        print('jjjj')
        #convert to a dataframe
        dataframe = pd.DataFrame.from_dict([j], orient='columns')
        dataframe.columns = asips_utils.DATAFRAME_HEAD
        print(dataframe)

        #pass dataframe
        log = self.make_stream_and_learn(dataframe)
        #delete from predictions collection in the firebase
        asips_utils.FIREBASE_REF.delete(asips_utils.BASE_PATH + asips_utils.PREDICTIONS_PATH, id)
        return log


    def make_stream_and_learn(self, dataframe):
        # Preprocess dataframe to create a stream
        dataframe = dataframe.drop(columns='y_pred')
        dataframe = dataframe.dropna()
        dataframe['target_class'] = dataframe['target_class'].astype(int)
        print(dataframe.head())

        # Make stream
        stream = DataStream(allow_nan=True, data=dataframe)
        self.print_stream_info(stream)

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
            result = asips_utils.FIREBASE_REF.post_async(asips_utils.BASE_PATH + asips_utils.CONFIRMED_PATH, data)
        end_time = datetime.datetime.now()
        time_difference = end_time - start_time

        string = 'Time difference: {0} \ninfo: {1} \nNumber of learning samples: {2}'
        log = string.format(time_difference, self.efdtgh.get_info,
                            len(dataframe.index))
        print(log)
        return log


