from flask import Flask, jsonify, request
from asips_delayed_progressive import Asips
app = Flask(__name__)

asips = Asips()
@app.route('/pretrain')
def pretrain():
    log = asips.pretrain()
    return log


@app.route('/predict')
def make_prediction():
    log = asips.make_prediction(asips.stream_first)
    return log

@app.route('/learn-from-all')
def learn_from_all():
    log = asips.learn_from_all()
    return log

@app.route('/learn/<id>')
def learn(id):
    log = asips.learn(id)
    return log

if __name__ == "__main__":
    app.run(debug=True)