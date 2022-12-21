# Chest Xray Image Prediction

## Author
Daniel Egbo

## Problem
Using chest xray images, this project attempts to train a convolutional neural networks (CNN) that categorises Chest Xray images into three disticnt classes: `Covid, Pneumonia and Normal.`

## Solution
The CNN model is trained using TensorFlow Keras Sequential API.

## Data
The dataset used for this project is from 



### Running Docker Image

```bash
docker run -it --rm \
  -p 8500:8500 \
  -v $(pwd)/chestxray-model:/models/chestxray-model/1 \
  -e MODEL_NAME="chestxray-model" \
  tensorflow/serving:2.7.0
```
Or if you're using Apple M1 (silicon) series, use the command:

```bash
docker run -it --rm \
  -p 8500:8500 \
  -v $(pwd)/chestxray-model:/models/chestxray-model/1 \
  -e MODEL_NAME="chestxray-model" \
  emacski/tensorflow-serving:latest-linux_arm64
```


```bash
docker build -t xray-model:v1 \
  -f image-model.dockerfile .
```

### Run the image
```bash
docker run -it --rm \
  -p 8500:8500 \
  xray-model:v1
```

```
docker build -t xray-gateway:001 \
  -f image-gateway.dockerfile .
```

```bash
docker run -it --rm \
  -p 9696:9696 \
  xray-gateway:001
```


pipenv run python gateway.py

```
docker-compose up -d