from flask import Flask, request, Markup, abort, jsonify, send_from_directory
from flask_cors import CORS
import logging
from logging.config import dictConfig
import sys, os
import base64
import traceback

DATA_ROOT = "./"
dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'DEBUG',
        'handlers': ['wsgi']
    }
})


app = Flask(__name__)
@app.route("/<path:path>")
def static_dir(path):
    return send_from_directory("../docs", path)

@app.route('/', methods=['GET'])
def redirect_to_index():
    return send_from_directory("../frontend/dist", 'index.html')

CORS(app, resources={r"/*": {"origins": "*"}}) 


## !!!!!!!!!!! COLABのプロキシがRoot直下のパスしか通さない??? !!!!!!
## !!!!!!!!!!! Bodyで参照、設定コマンドを代替する。 !!!!!!
@app.route('/api', methods=['POST'])
def api():
    try:
        print("POST REQUEST PROCESSING....\n")
        command = request.json['command']
        print(request)

        # GET VOICE
        if command == "GET_VOICE":
            title = request.json['title']
            prefix = request.json['prefix']
            index = request.json['index']

            data_dir = os.path.join(DATA_ROOT, title)
            os.makedirs(data_dir,exist_ok=True)

            filename = f"{prefix}{index:03}.zip"
            fullpath = os.path.join(data_dir, filename)

            is_file = os.path.isfile(fullpath)
            if is_file == False:
                data = {
                    "message":"NOT_FOUND",
                }
                return jsonify(data)

            f = open(fullpath, 'rb')
            data = f.read()
            dataBase64 = base64.b64encode(data).decode('utf-8')
            data = {
                "message":"OK",
                "data":dataBase64,
            }
            return jsonify(data)


        # POST VOICE
        if command == "POST_VOICE":
            title = request.json['title']
            prefix = request.json['prefix']
            index = request.json['index']
            data = base64.b64decode(request.json['data'])

            data_dir = os.path.join(DATA_ROOT, title)
            os.makedirs(data_dir,exist_ok=True)

            filename = f"{prefix}{index:03}.zip"
            fullpath = os.path.join(data_dir, filename)
            f = open(fullpath, 'wb')
            f.write(data)
            f.close()
            data = {
                "message":"OK_TEST"
            }
            return jsonify(data)
    except Exception as e:
        print("REQUEST PROCESSING!!!! EXCEPTION!!!", e)
        print(traceback.format_exc())
        return str(e)


## !!!!!!!!!!! COLABのプロキシがRoot直下のパスしか通さない??? !!!!!!
## !!!!!!!!!!! Bodyで参照、設定コマンドを代替する。 !!!!!!
# @app.route('/api/voice/<string:title>/<string:prefix>/<int:index>', methods=['POST'])
# def post_voice(title, prefix, index):
#     try:
#         filename = f"{prefix}{index:03}.zip"
#         data_dir = os.path.join(DATA_ROOT, title)
#         os.makedirs(data_dir,exist_ok=True)
#         fullpath = os.path.join(data_dir, filename)
#         data = base64.b64decode(request.json['data'])
#         f = open(fullpath, 'wb')
#         f.write(data)
#         f.close()
#         data = {
#             "message":"OK"
#         }
#         return jsonify(data)
#     except Exception as e:
#         print("REQUEST PROCESSING!!!! EXCEPTION!!!", e)
#         print(traceback.format_exc())
#         return str(e)

# @app.route('/api/voice/<string:title>/<string:prefix>/<int:index>', methods=['GET'])
# def get_voice(title, prefix, index):
#     filename = f"{prefix}{index:03}.zip"
#     data_dir = os.path.join(DATA_ROOT, title)
#     fullpath = os.path.join(data_dir, filename)

#     is_file = os.path.isfile(fullpath)
#     if is_file == False:
#         data = {
#             "message":"NOT_FOUND",
#         }
#         return jsonify(data)

#     f = open(fullpath, 'rb')
#     data = f.read()
#     dataBase64 = base64.b64encode(data).decode('utf-8')
#     data = {
#         "message":"OK",
#         "data":dataBase64,
#     }
#     return jsonify(data)



if __name__ == '__main__':
    args = sys.argv
    PORT = args[1]
    DATA_ROOT = args[2]
    app.logger.info('START APP')
    app.run(debug=True, host='0.0.0.0',port=PORT)
