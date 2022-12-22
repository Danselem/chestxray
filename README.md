# Chest Xray Image Prediction


## Problem
Using chest xray images, this project attempts to train a convolutional neural networks (CNN) that categorises Chest Xray images into three disticnt classes: `Covid, Pneumonia and Normal.`

## Solution
The CNN model is trained using TensorFlow Keras Sequential API. The model will take aim chest xray image and try to classify it into one of the following classes:

```zsh
- Covid
- Pneumonia and 
- Normal.
```

## Data
The dataset used for this project is from [Kaggle](https://www.kaggle.com/datasets/subhankarsen/novel-covid19-chestxray-repository?select=data).

## Setup
Clone the repository with the command:
```bash
git clone git@github.com:Danselem/chestxray.git
cd chestxray
```

The model was built using `Tensorflow 2.10.0.`

First, download recipes with the command:

```bash
python download_recipes.py
```
This will download the following:
- the image data repository `data`, 
- `chestxray-model` repository for tensorflow-serving,
- `xray_v1_15_0.913.h5` trained CNN model with validation accuracy of 91.3%.
- `xray_model.tflite`, a tensorflow-lite version of `xray_v1_15_0.913.h5.`

## Produce the model
If you're interested in reproducing the model with your tensorflow enabled GPU, use the command:
```bash
python train.py
```
Ensure `train.py` and the `data` directory are on the same project directory.

You can also run the `notebook.ipynb` to reproduce the same result.



### Buidling and Running Docker Image

<!-- ```bash
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
``` -->
To build a tensorflow-serving docker image with the chestxray-model, use the dockerfile `image-model.dockerfile` with the command:

```bash
docker build -t xray-model:v1 \
  -f image-model.dockerfile .
```
Please note that `image-model.dockerfile` can be modified depending on your PC architecture, either Apple M1 or not.

- For non Apple M series, ensure you edit `image-model.dockerfile` and uncomment `#FROM tensorflow/serving:2.7.0` and comment `FROM emacski/tensorflow-serving:latest-linux_arm64`
- Apple M series users can run the `image-model.dockerfile` without modifying it.

Next, you can build the image gateway with the command:
<!-- ```bash
docker run -it --rm \
  -p 8500:8500 \
  xray-model:v1 -->
<!-- ``` -->


```bash
docker build -t xray-gateway:001 \
  -f image-gateway.dockerfile .
```

<!-- ```bash
docker run -it --rm \
  -p 9696:9696 \
  xray-gateway:001
``` -->

## Docker Compose
After building both images, you can run the images using `docker-compose` with the command in detached mode:
<!-- pipenv run python gateway.py -->

```
docker-compose up -d
```

While in your terminal inside the project directory run the command:
```bash
pipenv install
```
to install the dependencies for testing the model.

```bash
pipenv run python test.py
```
This takes a chest xray image from a url, transform the image into tensors and parse it to the tensorflow model for prediction and ouputs the prediction results into any of the categories and the confidence value.

To stop the `docker-compose`, use the command:
```bash
docker-compose down
```

See the video below for how to run the docker compose in your environment.

<video src='https://cloudcape.saao.ac.za/index.php/s/JILbWNrXsvhNwX0/download'></video>


## License
Distributed under the terms of the [MIT](https://opensource.org/licenses/MIT) license, `chestxray` is free and open source software.