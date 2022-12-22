import os

# downloading data
os.system('wget https://cloudcape.saao.ac.za/index.php/s/Zx3E3QDS7zNk7gP/download -O chest_xray.zip')

# unzip data
os.system('unzip chest_xray.zip')

# chestxray-model.zip
os.system('wget https://cloudcape.saao.ac.za/index.php/s/1n9HFKrnop2kXca/download -O chestxray-model.zip')

# unzip chestxray-model.zip
os.system('unzip chestxray-model.zip')

# downloading xray_v1_15_0.913.h5
os.system('wget https://cloudcape.saao.ac.za/index.php/s/kRWAYgY43XaI3x8/download -O xray_v1_15_0.913.h5')

# downloading xray_model.tflite
os.system('wget https://cloudcape.saao.ac.za/index.php/s/ZwEzU9EvgW691lc/download -O xray_model.tflite')