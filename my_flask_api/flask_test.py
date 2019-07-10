from flask import Flask
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
@app.route('/')
def index():
    return "Index API"
if __name__ == '__main__':
    # This is used when running locally.
    app.run(host='0.0.0.0', debug=True)