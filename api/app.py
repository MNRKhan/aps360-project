# <!-- Template from https://github.com/floydhub/colornet-template -->

from flask import Flask, jsonify, request, send_file, make_response, render_template
from inference import *
import os
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename
from io import BytesIO
import base64
import unicodedata
from werkzeug.urls import url_quote

app = Flask(__name__)

# floyd run --data yulmart/datasets/weights/1:weights --mode serve

@app.route('/', methods=['GET'])
def index():
    return render_template('serving_template.html')

@app.route('/image', methods=["POST"])
def eval_image():

    input_file = request.files.get('file')
    if not input_file:
        return BadRequest("File not present in request")
    if input_file.filename == '':
        return BadRequest("File name is not present in request")
    if not input_file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        return BadRequest("Invalid file type")

    # # Save Image to process
    input_buffer = BytesIO()
    output_buffer = BytesIO()
    input_file.save(input_buffer)

    img = extract(input_buffer)
    img.save(output_buffer, format="PNG")
    img_str = base64.b64encode(output_buffer.getvalue())

    content_disposition = "attachment; filename=output.png"

    response = make_response(img_str)
    response.headers.set('Content-Type', 'image/png')
    response.headers.set('Content-Disposition', content_disposition)

    return response

    ##############

if __name__ == "__main__":
    print("* Starting web server... please wait until server has fully started")
    app.run(host='0.0.0.0', threaded=False)
