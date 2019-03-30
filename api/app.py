from flask import Flask, jsonify, request, send_file
from inference import *
import os
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route('/', methods=["POST"])
def evaluate():


    # File integrity (from FloydHub tutorial on style transfer) -----

    input_file = request.files.get('file')
    if not input_file:
        return BadRequest("File not present in request")

    filename = secure_filename(input_file.filename)
    if filename == '':
        return BadRequest("File name is not present in request")

    input_filepath = os.path.join(filename) # './images/',
    output_filepath = os.path.join(filename) # '/output/',
    input_file.save(input_filepath)

    # ---------------------------------------------------------------

    extract(input_filepath, output_filepath)

    return send_file(output_filepath, mimetype='image/jpg')

'''
if __name__ == "__main__":
    print("* Starting web server... please wait until server has fully started")
    app.run(host='0.0.0.0', threaded=False)
'''