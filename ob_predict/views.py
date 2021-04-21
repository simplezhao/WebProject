from django.shortcuts import render
from django.views.decorators.http import require_http_methods
import time
import json
import logger
import base64

import gunicorn
from flask import Flask, request, jsonify, make_response
from predictor import Predictor

# Create your views here.
@require_http_methods(['GET'])  # Health check URL for consul
def check():
    return "success"

"""config path"""
_CONF_PATH = "conf/conf.json"

# load model
predictor_instance = Predictor()
@require_http_methods(['POST'])
def inference(request):
    """app entry"""
    start = time.time()

    json_data = request.get_data()
    json_info = json.loads(json_data.decode("utf-8"))
    traceId = json_info.get("traceId")
    mode = json_info.get("mode")
    od_pair = json_info.get("od_pair")
    
    status = 0
    response = {}
    response["errorMsg"] = ""
    try:
        od_pair = tuple(od_pair.split('-'))
        predictor_instance.do_predict(od_pair, mode=mode)
    except Exception as e:
        response["errorMsg"] = "sorry, unknown exception, server failed."
        status = -1
        logger.error('Failed to execute process function, traceId:{}, error:{}'.format(traceId, e))

    response['traceId'] = traceId
    response['status'] = status

    response = make_response(
        jsonify(response),
        200, # status
        {'content-type': 'application/json'}
    )

    end = time.time()
    logger.info('traceId: ' + traceId + '\t' + 'monitor TotalTime' + '\t' + str((end-start)*1000) + 'ms')

    return response
