import os
import sys
import io
import time

from flask import Flask, request, redirect, url_for, flash, render_template_string
from flask import send_from_directory, send_file
from flask.json import jsonify
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from mini_detect import label_image, load_model, detect, get_meta
# from prometheus_flask_exporter import PrometheusMetrics
import torch


# Some constants
UPLOAD_FOLDER = './server/uploads'
DETECTIONS_FOLDER = './server/detections'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
# PRELOADED_MODELS = ["s", "x"]
PRELOADED_MODELS = ['yolov7', 'yolov7x']

# This should be an external file
FORM_TEMPLATE = '''
<!doctype html>
<head>
<title>YOLOv7 - Subir imagen</title>
<style>
ul.flashes {
    background-color: #f0d0d0;
    color: #804040;
    list-style-type: none;
    width: 25%;
}
</style>
</head>
<body>
<h1>YOLOv7 - Subir imagen</h1>
{% with messages = get_flashed_messages() %}
{% if messages %}
    <ul class=flashes>
    {% for message in messages %}
    <li>{{ message }}</li>
    {% endfor %}
    </ul>
{% endif %}
{% endwith %}        
<form method=post enctype=multipart/form-data target=_blank>
    <input type="file" name="file" id="file">
    <input type=submit value=Subir>

  <label for="model">Elige modelo:</label>
  <select name="model" id="model">
    <option value="yolov7">Yolov7</option>
    <option value="yolov7x">Yolov7x</option>
  </select>

  <label for="threads">Num Threads:</label>
  <input type="number" id=threads name=threads min='0'>
  
    <input type=checkbox id=augment name=augment value=augment>
    <label for=augment>Augment</label>

    <p>Tipo de resultado</p>
    <input type=radio id=imagen name=tipo value=imagen checked>
    <label for=imagen>Imagen etiquetada</label>
    <input type=radio id=json name=tipo value=json> 
    <label for=json>JSON</label>

</form>
</body>
'''

auto_numthreads = None

def autoset_threads():
    global auto_numthreads
    if auto_numthreads is None:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            quota = int(f.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            period = int(f.read())
        if quota == -1:
            return
        auto_numthreads = max(1, quota//period)
    torch.set_num_threads(auto_numthreads)
    # print(f"Setting num_threads to {num_threads}", file=sys.stderr)

autoset_threads()

# Create app
app = Flask(__name__)
# metrics = PrometheusMetrics(app)
# metrics.info("yolo_info", "YoloV5 application info", version="1.7s")

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DETECTIONS_FOLDER'] = DETECTIONS_FOLDER
app.config['SECRET_KEY'] = "replace-me"

model = {}
# Prepare Yolo models
for m in PRELOADED_MODELS:
    print(f"Loading model {m}")
    model[m] = load_model(f"{m}")

# Utility functions
def allowed_file(filename):
    """Checks if the extension of the file sent by the client is valid"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_file(request):
    """Extracts the file from the client requests and saves it
    in the UPLOAD_FOLDER.

    Returns the filename in the filesystem if all went ok,
    or an empty string otherwise
    """

    if 'file' not in request.files or not request.files["file"]:
        return ""
    file = request.files['file']
    if not file or not allowed_file(file.filename):
        return ""
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return filename


def detect_and_label_image(file, m, augment):
    """Calls Yolo detector on the input image and produces
    a labelled image in the output folder"""

    if type(file) == str:
        _in = os.path.join(app.config['UPLOAD_FOLDER'], file)
        _out = app.config['DETECTIONS_FOLDER']
    else:
        _in = cv2.imdecode(np.asarray(bytearray(file.read()), dtype=np.uint8),1)
        _out = None
    r = detect(_in, model[m], augment=augment)
    img = label_image(_in, r, _out)
    return img



def detect_and_get_meta(file, m, augment):
    """Calls Yolo detector on the input image and returns
    a Python dictionary with the results"""

    if type(file) == str:
        _in = os.path.join(app.config['UPLOAD_FOLDER'], file)
    else:
        _in = cv2.imdecode(np.asarray(bytearray(file.read()), dtype=np.uint8),1)
    r = detect(_in, model[m], augment=augment)
    return get_meta(r)


# FLASK routes
@app.route('/detect2', methods=['GET', 'POST'])
def upload_file():
    """Main route. When GET is used it shows the form which allows
    to upload a file. When POST is used, it receives the file, calls
    Yolo detector and returns the result, which can be an annotated
    image or a JSON
    """

    t1 = time.time()
    if request.method == "GET":
        return render_template_string(FORM_TEMPLATE)

    # Dealing with POST request:
    filename = get_file(request)
    if not filename:
        flash("Tipo de fichero no válido (solo jpg o png)")
        return redirect(request.url)

    m = request.form.get("model", "yolov7")
    augment = request.form.get("augment", "false")
    if augment.lower() in ("false", "0", "no"):
       augment = False
    augment = bool(augment)
    
    # Return labeled image
    if request.form["tipo"] == "imagen":
        detect_and_label_image(filename, m, augment)
        # Redirect to the url which allows the client to
        # download the annotated image
        return redirect(url_for('uploaded_file', augment))
    # Return json
    else:
        metadata = detect_and_get_meta(filename, m, augment)
        t2 = time.time()
        metadata["meta"]["totaltime"] = t2-t1
        metadata["meta"]["model"] = m            
        return jsonify(metadata)

@app.route('/detect', methods=['GET', 'POST', 'HEAD'])
# @metrics.do_not_track()
# @metrics.summary('requests_by_status', 'Request latencies by status',
#                 labels={'status': lambda r: r.status_code})
# @metrics.summary('requests_by_type', 'Request latencias by type',
#		labels={'type': lambda: request.method=="GET" or request.form["tipo"] })
# @metrics.gauge('in_progress', 'Long running requests in progress')
# @metrics.histogram('requests_by_status_and_type', 'Request latencies by status and type',
#                   labels={'status': lambda r: r.status_code, 'type': lambda: request.form["tipo"]})
def direct_detect():
    """When GET is used it shows the form which allows
    to upload a file. When POST is used, it receives the file, calls
    Yolo detector and returns the result, which can be an annotated
    image or a JSON.
    This is done without using the filesystem
    """
    t1 = time.time()

    if request.method == "HEAD":
        return ""
    if request.method == "GET":
        return render_template_string(FORM_TEMPLATE)

    # Dealing with POST request:
    print("Recibido", request.method, dict(request.form), dict(request.files))
    file = request.files.get("file")

    m = request.form.get("model", "yolov7")
    if m not in PRELOADED_MODELS:
        return f"You cannot use model {m!r}, only {PRELOADED_MODELS}", 404
    augment = request.form.get("augment", "false")
    if augment.lower() in ("false", "0", "no"):
       augment = False
    augment = bool(augment)

    threads = request.form.get("threads", "auto")
    try:
        threads = int(threads)
    except:
        threads = 0
    if threads <= 0:
        autoset_threads()
    else:
        torch.set_num_threads(threads)

    if request.form["tipo"] == "imagen":
        img = detect_and_label_image(file.stream, m, augment)
        return send_file(
            io.BytesIO(img),
            # attachment_filename="detection.jpg",
            mimetype="image/jpg"
        )
    else:
        metadata = detect_and_get_meta(file.stream, m, augment)
        t2 = time.time()
        metadata["meta"]["totaltime"] = t2-t1
        metadata["meta"]["model"] = m
        metadata["meta"]["threads"] = torch.get_num_threads()
        metadata["meta"]["inter_op"] = torch.get_num_interop_threads()
        return jsonify(metadata)

# Route to "return" the annotated image
@app.route('/detections/<filename>')
# @metrics.do_not_track()
# @metrics.summary('downloads_by_status', 'Download latencies by status',
#                 labels={'status': lambda r: r.status_code})
def uploaded_file(filename):
    return send_from_directory(app.config['DETECTIONS_FOLDER'],
                               filename)
