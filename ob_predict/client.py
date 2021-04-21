# -*- coding: utf-8 -*-
# File: client.py

"""
example to call server. 
"""

import requests
import json
import time
import base64


class HttpClient():
    def __init__(self, host, port, appname):
        self.host = host
        self.port = port
        self.appname = appname

    def request(self, request_id, od_pair, mode):
        url = "http://{}:{}/{}".format(self.host, self.port, self.appname)
        params = {
            "traceId": request_id,
            "mode": mode,
            "od_pair": od_pair
        }
        headers = {
            'Content-Type': 'application/json'
        }
        res = requests.post(url, json=params, headers=headers)
        res = res.text.encode("utf-8")
        res = json.loads(res)
        return res

if __name__ == "__main__":
    client=HttpClient("0.0.0.0", "6008", "kalman-od-predict")
    client.request(request_id="test", od_pair="Sta134-Sta63", mode="predict_future")
