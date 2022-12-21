#!/usr/bin/env python
# coding: utf-8

import os
import grpc

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

import numpy as np

# from keras_image_helper import create_preprocessor

from flask import Flask
from flask import request
from flask import jsonify
from PIL import Image
import requests

from proto import np_to_protobuf

host = os.getenv('TF_SERVING_HOST', 'localhost:8500')

channel = grpc.insecure_channel(host)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

# preprocessor = create_preprocessor('xception', target_size=(299, 299))


def preprocessor(url):
    img = Image.open(requests.get(url, stream=True).raw)

    img = img.resize((299, 299), Image.Resampling.NEAREST)


    x = np.array(img, dtype='float32')
    X = np.array([x])

    return X

def prepare_request(X):
    pb_request = predict_pb2.PredictRequest()

    pb_request.model_spec.name = 'chestxray-model'
    pb_request.model_spec.signature_name = 'serving_default'

    pb_request.inputs['sequential_input'].CopyFrom(np_to_protobuf(X))
    return pb_request


classes = ['Covid', 'Normal', 'Pneumonia']

def softmax(x):
    return(np.exp(x)/np.exp(x).sum())

def prepare_response(pb_response):
    preds = pb_response.outputs['dense_1'].float_val
    score_lite = softmax(preds)
    feedback = "This image most likely belongs to {} with a {:.2f} percent confidence."
    
    return dict(zip(classes, preds)), feedback.format(classes[np.argmax(score_lite)], 100 * np.max(score_lite))


def predict(url):
    X = preprocessor(url)
    pb_request = prepare_request(X)
    pb_response = stub.Predict(pb_request, timeout=20.0)
    response = prepare_response(pb_response)
    return response


app = Flask('gateway')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json()
    url = data['url']
    result = predict(url)
    return jsonify(result)


if __name__ == '__main__':
    # url = 'https://cloudcape.saao.ac.za/index.php/s/u5CJ4KfGKICKfDS/download'
    # response, feedback = predict(url)
    # print(response)
    # print(feedback)
    app.run(debug=True, host='0.0.0.0', port=9696)

