from firebase import firebase
import asips_utils
from firebase import jsonutil
import json
import pandas as pd

class FirebaseHelper:

    def __init__(self):
        pass

    @staticmethod
    def save_pretrain_info(time_difference, efdtgh, n_samples, stream_pre):
        pretrain_info_path = asips_utils.BASE_PATH + asips_utils.PRETRAIN_INFO_PATH
        asips_utils.FIREBASE_REF.put_async(pretrain_info_path, "timeDifference", time_difference)
        asips_utils.FIREBASE_REF.put_async(pretrain_info_path, "info", str(efdtgh.get_info))
        asips_utils.FIREBASE_REF.put_async(pretrain_info_path, "numberOfLearningSamples", str(n_samples))
        asips_utils.FIREBASE_REF.put_async(pretrain_info_path, "features", stream_pre.feature_names)
        asips_utils.FIREBASE_REF.put_async(pretrain_info_path, "targetValues", str(stream_pre.target_values))

    @staticmethod
    def save_current_mean_measures(dict):
        dict['confirmationState'] = 0
        asips_utils.FIREBASE_REF.put_async(asips_utils.BASE_PATH, asips_utils.CURRENT_MEAN_EVALUATION_M_PATH, dict)

    @staticmethod
    def save_current_evaluation_measures(current_measures):
        eval_measures_path = asips_utils.BASE_PATH + asips_utils.CURRENT_EVALUATION_MEASURES_PATH
        asips_utils.FIREBASE_REF.put_async(eval_measures_path, "kappa", current_measures.kappa_score())
        asips_utils.FIREBASE_REF.put_async(eval_measures_path, "gMean", current_measures.geometric_mean_score())
        asips_utils.FIREBASE_REF.put_async(eval_measures_path, "accuracy", current_measures.accuracy_score())
        asips_utils.FIREBASE_REF.put_async(eval_measures_path, "recall", current_measures.recall_score())
        asips_utils.FIREBASE_REF.put_async(eval_measures_path, "precision", current_measures.precision_score())
        asips_utils.FIREBASE_REF.put_async(eval_measures_path, "fScore", current_measures.f1_score())
        asips_utils.FIREBASE_REF.put_async(eval_measures_path, "samples", current_measures.n_samples)
        asips_utils.FIREBASE_REF.put_async(eval_measures_path, "confirmationState", 0)

    @staticmethod
    def save_kappa_values(current_measures,dict):
        mean_performance_path = asips_utils.BASE_PATH + asips_utils.MEAN_PERFORMANCES_PATH

        kappa_count = asips_utils.FIREBASE_REF.get(mean_performance_path, 'kappa')
        asips_utils.FIREBASE_REF.put(mean_performance_path + 'kappa', len(kappa_count), current_measures.kappa_score())

        m_kappa_count = asips_utils.FIREBASE_REF.get(mean_performance_path, 'meanKappa')
        asips_utils.FIREBASE_REF.put(mean_performance_path + 'meanKappa', len(m_kappa_count), dict['kappa'])

    @staticmethod
    def save_signal_to_firebase(collection_path, signal_data):
        asips_utils.FIREBASE_REF.post_async(asips_utils.BASE_PATH + collection_path, signal_data)

    @staticmethod
    def get_signal_from_firebase(id):
        return asips_utils.FIREBASE_REF.get(asips_utils.BASE_PATH + asips_utils.PREDICTIONS_PATH, id)

    @staticmethod
    def delete_signal_from_firebase(id):
        asips_utils.FIREBASE_REF.delete(asips_utils.BASE_PATH + asips_utils.PREDICTIONS_PATH, id)

    @staticmethod
    def update_signal_record(id, model, y_pred):
        asips_utils.FIREBASE_REF.put( asips_utils.BASE_PATH + asips_utils.PREDICTIONS_PATH + '/' +id, model, y_pred)
