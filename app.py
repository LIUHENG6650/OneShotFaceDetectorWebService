from flask import Flask
from flask import jsonify, request
from configuration import CONFIG
from FaceDBModel import DBModel
from FaceDetector import FaceDetector
from EnhancedFacenet import EnhancedFacenet
from annoy import AnnoyIndex
import os
import cv2
import numpy as np


detector = FaceDetector()
embedder = EnhancedFacenet()
app = Flask(__name__)
fdModel = DBModel()

#Load annoy index
annoyIndex = AnnoyIndex(512, metric='euclidean')

if os.path.isfile(CONFIG['annoy_filename']):
    annoyIndex.load(CONFIG['annoy_filename'])
else:
    for row in fdModel.getAllFaces():
        annoyIndex.add_item(row['id'], row['data'])

    annoyIndex.build(32)
    annoyIndex.save(CONFIG['annoy_filename'])
fdModel.close()

def response( data=[], message=None, error=None):
    result = {}
    result['data'] = data
    result['success'] = (error is None)
    if message is not None:
        result['message'] = message
    if error is not None:
        result['error'] = error
    return jsonify(result)

@app.route('/detect', methods=['POST'])
def detect():
    try:
        file = request.files['image']
        if not file.content_type == 'image/jpeg' and not file.content_type == 'image/png':
            return response(error="File extension should be png or jpeg.")
        filestring = file.read()
        img = np.fromstring(filestring, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ##Histogram equalization
        b, g, r = cv2.split(img)
        b = cv2.equalizeHist(b)
        g = cv2.equalizeHist(g)
        r = cv2.equalizeHist(r)
        img_he = cv2.merge((b,g,r))
        rects = detector.detect(img_he)

        if len(rects) == 0:
           return response(error="No face detected."), 400
        message = None
        if len(rects)>1:
            message = "There are more than one face, selecting the face with maximum area."
        max_rect = None
        max_area = 0
        for rect in rects:
            area = rect.width()*rect.height()
            if area>max_area:
                max_area = area
                max_rect = rect

        face, embedding = embedder.alignAndEncode(img, gray, max_rect)
        results, distances = annoyIndex.get_nns_by_vector(embedding, 1, include_distances=True)
        ##Threshold =
        threshold = CONFIG['default_threshold']
        try:
            threshold = request.form['threshold']
        except:
            pass
        if len(distances) == 0:
            return response()
        elif distances[0]>threshold:
            return response()
        else:
            fdModel = DBModel()
            person = fdModel.getFaceById(results[0])
            last_result = {}
            if len(person)>0:
                last_result = {"id": person[0]["face_id"], "distance": distances[0]}
            fdModel.close()
            return response(data=last_result)


    except Exception as e:
        print(e)
        return response(error="There is not any image in sent file or there is an error while reading."), 400

@app.route('/add', methods=['POST'])
def add():
    try:
        file = request.files['image']
        if not file.content_type == 'image/jpeg' and not file.content_type == 'image/png':
            return response(error="File extension should be png or jpeg.")
        filestring = file.read()
        img = np.fromstring(filestring, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ##Histogram equalization
        b, g, r = cv2.split(img)
        b = cv2.equalizeHist(b)
        g = cv2.equalizeHist(g)
        r = cv2.equalizeHist(r)
        img_he = cv2.merge((b,g,r))
        rects = detector.detect(img_he)

        if len(rects) == 0:
           return response(error="No face detected"), 400
        message = ""
        if len(rects)>1:
            message = "There are more than one face, selecting the face with maximum area."
        max_rect = None
        max_area = 0
        for rect in rects:
            area = rect.width()*rect.height()
            if area>max_area:
                max_area = area
                max_rect = rect
        #Face idyi al
        face_id = None
        try:
            face_id = request.form['name']
        except:
            return response(error="name field must be filled."), 400
        face, embedding = embedder.alignAndEncode(img, gray, max_rect)

        fdModel = DBModel()
        fdModel.addFace(face_id, embedding)
        fdModel.close()

        message += "Face added."
        return response(message=message)
    except Exception as e:
        print(e)
        return response(error="There is not any image in sent file or there is an error while reading."), 400

    return 'Add'

@app.route('/balance', methods=['POST'])
def balance():
    fdModel = DBModel()
    try:
        global annoyIndex
        annoyIndex.unload()
        annoyIndex = AnnoyIndex(512, metric='euclidean')
        for row in fdModel.getAllFaces():
            annoyIndex.add_item(row['id'], row['face_data'])
        fdModel.close()
        annoyIndex.build(32)
        annoyIndex.save(CONFIG['annoy_filename'])
    except:
        return  response(error="There is an error."), 500

    return response(message="Indexing done.")

@app.route('/delete', methods=['POST'])
def delete():
    face_id = None
    try:
        face_id = request.form['name']
    except:
        return response(error="name field must be filled."), 400
    fdModel = DBModel()
    fdModel.deleteFaceId(face_id)
    fdModel.close()
    return response(message="Deleted.")

@app.route('/clean', methods=['POST'])
def clean():
    fdModel = DBModel()
    fdModel.truncate()
    balance()
    return response(message="Face database cleaned.")

if __name__ == '__main__':
    app.run()
