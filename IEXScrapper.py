import sys
import requests
import json
from pymongo import MongoClient
from pprint import pprint

client = MongoClient('mongodb://localhost:27017')
db = client.local

for stock in sys.argv[1:]:
  col = db[stock]

  response = requests.get('https://api.iextrading.com/1.0/stock/' + stock + '/chart/1d')

  data_dict = json.loads(response.content)

  for val in data_dict:
    col.insert_one(val)





