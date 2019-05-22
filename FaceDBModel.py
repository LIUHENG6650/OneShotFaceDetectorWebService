from configuration import CONFIG
import sqlite3
import numpy as np



class DBModel:
    def __init__(self):
        self.conn = sqlite3.connect(CONFIG['sqlite_filename'])
        self.curs = self.conn.cursor()

    def truncate(self):
        self.curs.execute('DELETE FROM faces')
        self.conn.commit()
        return True

    def addFace(self, face_id, face_data):
        data = np.array(face_data).astype(dtype=np.float32).tostring()
        self.curs.execute('INSERT INTO faces(face_id, face_data) VALUES(?, ?)', (face_id, data))
        self.conn.commit()
        return self.curs.lastrowid
    def deleteFaceId(self, face_id):
        self.curs.execute('DELETE FROM faces WHERE face_id=?', [face_id])
        self.conn.commit()
    def getFaceByFaceId(self, face_id):
        self.curs.execute('SELECT id, face_id, face_data FROM faces WHERE face_id = ?', [face_id])
        resultset = []
        for row in self.curs.fetchall():
            face_data = np.fromstring(row[2], dtype=np.float32)
            resultset.append({'id': row[0], 'face_id': row[1], 'face_data': face_data})
        return resultset
    def getFaceById(self, id):
        self.curs.execute('SELECT id, face_id, face_data FROM faces WHERE id = ?', [id])
        resultset = []
        for row in self.curs.fetchall():
            face_data = np.fromstring(row[2], dtype=np.float32)
            resultset.append({'id': row[0], 'face_id': row[1], 'face_data': face_data})
        return resultset
    def getAllFaces(self):
        self.curs.execute('SELECT id, face_id, face_data FROM faces')
        resultset = []
        for row in self.curs.fetchall():
            face_data = np.fromstring(row[2], dtype=np.float32)
            resultset.append({'id': row[0], 'face_id': row[1], 'face_data': face_data})
        return resultset

    def close(self):
        self.curs.close()
        self.conn.close()




