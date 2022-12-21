import requests

url = 'http://localhost:9696/predict'

# url = 'http://localhost:8080/predict'

data = {'url': 'https://cloudcape.saao.ac.za/index.php/s/u5CJ4KfGKICKfDS/download'}

result = requests.post(url, json=data).json()
print(result)