import dlib
from imutils.face_utils.facealigner import FaceAligner
import facenet
import cv2
import tensorflow as tf


class Encoder:
    def __init__(self, checkpoint="models/20180402-114759"):
        self.sess = tf.Session()
        with self.sess.as_default():
            facenet.load_model(checkpoint)

    def generate_embedding(self, face):
        # Get input and output tensors
        images_placeholder = self.sess.graph.get_tensor_by_name("input:0")
        embeddings = self.sess.graph.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = self.sess.graph.get_tensor_by_name("phase_train:0")

        prewhiten_face = facenet.prewhiten(face)

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
        return self.sess.run(embeddings, feed_dict=feed_dict)[0]


class EnhancedFacenet:
    def __init__(self,desiredFaceWidth=160, predictor_path='models/shape_predictor_68_face_landmarks.dat', ):
        self.predictor = dlib.shape_predictor(predictor_path)
        self.fa = FaceAligner(self.predictor, desiredFaceWidth=desiredFaceWidth, desiredLeftEye=(0.32, 0.35))
        self.encoder = Encoder()


    def alignAndEncode(self, img, gray, face_rect):
        face = self.fa.align(img, gray, face_rect )
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        return face, self.encoder.generate_embedding(face_rgb)






if __name__ == '__main__':
    pass





#predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
#fa = FaceAligner(predictor, desiredFaceWidth=200)
#
#detector = dlib.get_frontal_face_detector()
#
#
#vs = cv2.VideoCapture(0)
#
##frame = cv2.imread()
#aligned = np.zeros((120,120,3)).astype(np.uint8)
#print(aligned)
#while True:
#    det, frame = vs.read()
#    print(type(frame))
#    if det:
#        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#        D=detector(gray, 1)
#        for a in D:
#            x1, y1 = a.left(), a.top()
#            x2, y2 = x1 + a.width(), y1 + a.height()
#
#            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), 1)
#            aligned = fa.align(frame, gray, a)
#
#        cv2.imshow("name", frame)
#        cv2.imshow("aligned", aligned)
#
#    cv2.waitKey(2)

