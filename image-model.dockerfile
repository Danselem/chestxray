#FROM tensorflow/serving:2.7.0
FROM emacski/tensorflow-serving:latest-linux_arm64


COPY chestxray-model /models/chestxray-model/1
ENV MODEL_NAME="chestxray-model"