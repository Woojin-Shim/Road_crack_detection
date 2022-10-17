import mimetypes
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, Response
import flask
import numpy as np
import cv2
import json
import os
import argparse
import datetime
import crack_detector

DEFAULT_PORT = 5000
DEFAULT_HOST = '0.0.0.0'
def parse_args():
    parser = argparse.ArgumentParser(description='Tensorflow object detection API')
    parser.add_argument('--debug', dest='debug',
                        help='Run in debug mode.',
                        required=False, action='store_true', default=True)
    parser.add_argument('--port', dest='port',
                        help='Port to run on.', type=int,
                        required=False, default=DEFAULT_PORT)
    parser.add_argument('--host', dest='host',
                        help='Host to run on, set to 0.0.0.0 for remote access', type=str,
                        required=False, default=DEFAULT_HOST)
    args = parser.parse_args()
    return args

# Initialize the Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])


#웹 서비스(crack)
@app.route('/crack', methods=['GET', 'POST']) ## 바꿔야할거
def index_road():
    return render_template('index_test.html') ## 바꿔야할 거

#crack 처리  
@app.route('/crack_process', methods=['POST']) ## 바꿀꺼
def crack_process():
    file = request.files['file']
    nparr = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    filename_first = datetime.datetime.now ().strftime('%y%m%d_%H%M%S')
    det = crack_detector.detect(image) ## 바꿀꺼
#     # dictection이 안 되었을 경우를 대비하여 None인지 체크
    if det is None:
        det = np.zeros((1, 6)) # 더미 데이터를 넣음!

    crack_labels = ['pothole', 'crack'] ## 바꿀꺼
    labels = [] # txt받기
    for _, _, _, _, _, cat in det:
        label_txt = crack_labels[int(cat)]
        labels.append(label_txt)

    #write json
    json_path = os.path.join('outputs', filename_first + '.json')
    result = {'rects': det[:,:4].tolist(), 'labels':labels, 'scores':det[:,-2].tolist()}
    json.dump(result, open(json_path, 'w'))

    #write image
    #drawing...
    dst = image
    dst = crack_detector.draw_boxes(dst, det) ## 바꿀꺼
    dst_path = os.path.join('outputs', filename_first + '.png')
    cv2.imwrite(dst_path, dst)
    return redirect(url_for('show_crack', filename_first=filename_first))

#outputs 폴더를 일반 웹서버 형식으로 오픈
@app.route('/outputs/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    output_path = os.path.join(app.root_path, 'outputs')
    return send_from_directory('outputs', filename)

#디텍션(crack) 결과 보여주기
@app.route('/crack/<filename_first>') ## 바꿀꺼
def show_crack(filename_first):
    json_path = os.path.join('outputs', filename_first + '.json')
    dst_path = os.path.join('..', 'outputs', filename_first + '.png')
    print(dst_path)
    result = json.load(open(json_path, 'r'))
    message_l = str(result['labels'])
    message_s = str(result['scores'])

    return render_template("result_test.html", image_path=dst_path, message=message_l, message_s=message_s)


# start flask app
def main():
    os.makedirs('outputs', exist_ok=True)
    args = parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)
if __name__ == "__main__":
    main()

app.run(debug=True)