import threading

from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
from datetime import datetime, timedelta
import json


class Tester:
    def validate(self, value, no_days):
        now = datetime.now()
        run_at = now + timedelta(days=no_days)
        delay = (run_at - now).total_seconds()
        threading.Timer(delay, self.test, [value])

    def test(self, date, predicted_value, erorr_rate=0.3):
        url = f'https://api.coinpaprika.com/v1/coins/btc-bitcoin/ohlcv/historical?start={date}&end={date}'
    
        headers = {
          'Accepts': 'application/json'
        }
    
        session = Session()
        session.headers.update(headers)
    
        try:
            response = session.get(url)
            data = json.loads(response.text)
            mean = (data[0]['low'] + data[0]['high'])/2
            if abs(predicted_value - mean) < erorr_rate*mean:
                return True
            else:
                return False
        
        except (ConnectionError, Timeout, TooManyRedirects) as e:
          print(e)
          return False
  

#EXEMPLU  
#in alt fisier .py se apeleaza asa

#2022-01-08 ziua pt care e facuta predictia
#41924 valoarea predictiei
#eroarea acceptata fata de media acelei zile (min+low)/2

#from tests import Tester
#
#testing = Tester()
#result = testing.test("2022-01-08", 41924, 0.5)
#if result:
#  print("Happy training!")
#else:
#  print("Sad sad sad!")

