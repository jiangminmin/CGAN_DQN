import os
import sys
import logging
import cv2
from flask import Flask, request, jsonify,json
from flask_cors import CORS
import struct
import numpy as np

from my_serve import get_model_api


# define the app
app = Flask(__name__)
CORS(app) # needed for cross-domain requests, allow everything by default


# logging for heroku
if 'DYNO' in os.environ:
    app.logger.addHandler(logging.StreamHandler(sys.stdout))
    app.logger.setLevel(logging.INFO)


# load the model


model_api = get_model_api()

# API route0
@app.route('/api', methods=['POST'])
def api():
    """API function

    All model-specific logic to be defined in the get_model_api()
    function
    """
    """input_data = request.files['file'].filename"""
    input_data = request.files['file']
    input_data = input_data.stream.read()
    input_data = np.fromstring(input_data,np.uint8)
    input_data = cv2.imdecode(input_data,cv2.IMREAD_GRAYSCALE)
    output_data = model_api(input_data,)
    cv2.imwrite("myoutput.jpg",output_data)
    response = jsonify(output_data.tolist())

    return response


@app.route('/')
def index():
    return "Index API"

# HTTP Errors handlers
@app.errorhandler(404)
def url_error(e):
    return """
    Wrong URL!
    <pre>{}</pre>""".format(e), 404


@app.errorhandler(500)
def server_error(e):
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    # This is used when running locally.
    app.run(host='0.0.0.0', debug=False)
