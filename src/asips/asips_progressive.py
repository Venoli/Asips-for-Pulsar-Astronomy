from skmultiflow.trees.extremely_fast_decision_tree import ExtremelyFastDecisionTreeClassifier
import datetime

start_time = datetime.datetime.now()
stream = FileStream('pulsar_data_train.csv')

efdtgh = ExtremelyFastDecisionTreeClassifier(split_criterion='gaussian_hellinger')

print("info: ")
print(efdtgh.get_info)
# Setup variables to control loop and track performance
n_samples = 0
max_samples = 9000
wait_samples = 300
y_true_all = []
y_pred_all = []
# Train the estimator with the samples provided by the data stream
while n_samples < max_samples and stream.has_more_samples():
    X, y = stream.next_sample()
    y_pred = efdtgh.predict(X)
    if (n_samples > wait_samples):
        y_true_all.append(y[0])
        y_pred_all.append(y_pred[0])
    n_samples += 1
    efdtgh.partial_fit(X, y)

end_time = datetime.datetime.now()
time_difference = end_time - start_time
print('{} samples analyzed.'.format(n_samples))
print('{} Time difference'.format(time_difference))
print("info: ")
print(efdtgh.get_info)