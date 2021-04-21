from firebase import firebase
FIREBASE_REF = firebase.FirebaseApplication('https://asips-3efdd-default-rtdb.firebaseio.com/', None)
BASE_PATH = 'asips-3efdd-default-rtdb/'
PREDICTIONS_PATH = 'Predictions'
CONFIRMED_PATH = 'Confirmed'
CURRENT_EVALUATION_MEASURES_PATH = 'CurrentEvaluationMeasures/'
MEAN_PERFORMANCES_PATH = 'MeanPerformances/'
CURRENT_MEAN_EVALUATION_M_PATH = 'CurrentMeanEvaluationMeasures/'
PRETRAIN_INFO_PATH = 'PretrainInfo/'
DATAFRAME_HEAD = ['Mean of the integrated profile', 'Standard deviation of the integrated profile',
                     'Excess kurtosis of the integrated profile', 'Skewness of the integrated profile',
                     'Mean of the DM-SNR curve', 'Standard deviation of the DM-SNR curve',
                     'Excess kurtosis of the DM-SNR curve', 'Skewness of the DM-SNR curve',
                     'target_class']