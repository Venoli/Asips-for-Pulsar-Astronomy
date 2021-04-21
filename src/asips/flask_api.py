from flask import Flask, jsonify, request
from asips_delayed_progressive import Asips
from test_with_other_classifiers import TestWithOtherClassifiers

app = Flask(__name__)

asips = Asips()
@app.route('/pretrain/<count>')
def pretrain(count):
    log = asips.pretrain(int(count))
    return {'message':log}


@app.route('/predict/<count>')
def make_prediction(count):
    log = asips.make_prediction(int(count))
    return {'message':log}

@app.route('/learn-from-all')
def learn_from_all():
    log = asips.learn_from_all()
    return {'message':log}

@app.route('/learn/<id>')
def learn(id):
    log = asips.learn(id)
    return {'message':log}

@app.route('/test-with-other-classifier/<model>')
def test_with_other_classifier(model):
    test_other = TestWithOtherClassifiers()
    log = test_other.classification_pipeline(model)
    return {'message':log}

if __name__ == "__main__":
    app.run(debug=True)