import os, sys, sqlite3
from PyQt5 import QtGui, QtCore, QtSql
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QDialog, QApplication, QMainWindow, QMessageBox, QInputDialog, QFileDialog, QPushButton, QDialogButtonBox, QTableWidgetItem
from PyQt5.uic import loadUi
import cv2
import numpy as np
from datetime import datetime
from PIL import Image, ImageDraw
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import load_img , img_to_array
from keras.preprocessing import image
from keras.models import model_from_json
from tensorflow.keras.models import model_from_json
import face_recognition
import dlib
import time
from datetime import datetime
from PyQt5.QtSql import QSqlDatabase



class USER(QDialog):        # Dialog box for entering name and key of new dataset.
    """USER Dialog """
    def __init__(self):
        super(USER, self).__init__()
        loadUi("GUI/showUsers.ui", self)


        self.source_path = ""
        self.name = ""
        self.extension = '.jpg'

        self.uploadphotoButton.clicked.connect(self.open_file)
        self.saveButton.clicked.connect(self.save)
        self.saveButton.clicked.connect(self.show_users)
        self.removeButton.clicked.connect(self.remove_users)
        self.listWidget.clicked.connect(self.preview_image)

        self.path = 'media/faces'


        self.show_users()

    def open_file(self):
        self.source_path = QFileDialog.getOpenFileName()
        self.source_path = self.source_path[0]
        thumbnail = QPixmap(self.source_path)
        thumbnail = thumbnail.scaled(120, 120)

        self.userPhoto.setPixmap(thumbnail)
        if self.source_path.endswith('.jpg') or self.source_path.endswith('.png'):
            file, self.extension  = self.source_path.split('.')
    
    def save(self):
        self.name = self.nameInput.text()
        if self.source_path != "":
            if self.name != "":
                img = cv2.imread(self.source_path)
                newPath = self.path + '/' + self.name + '.' + self.extension
                cv2.imwrite(newPath, img)
            else:
                QMessageBox().about(self, "Warning", "Bitte geben Sie einen Namen ein.")
        else:
            QMessageBox().about(self, "Warning", "Bitte fügen Sie ein Foto hinzu.")

    def show_users(self):
        files = [x for x in os.listdir(self.path)]
        for index, file in enumerate(files):
            if file.endswith('.jpg') or file.endswith('.png'):
                self.listWidget.takeItem(index)
                self.listWidget.insertItem(index, file)

    def remove_users(self):
        item = self.listWidget.currentItem()
        os.remove(self.path + '/' + item.text())
        QMessageBox().about(self, "Warning", "{} is gelöscht.".format(item.text()))

    def preview_image(self):
        item = self.listWidget.currentItem()
        path = self.path + '/' + item.text()
        thumbnail = QPixmap(path)
        thumbnail = thumbnail.scaled(120, 120)
        self.userPhoto.setPixmap(thumbnail)


    


class FaceApp(QMainWindow):        # Main application 
    """Main Class"""
    def __init__(self):
        super(FaceApp, self).__init__()
        loadUi("GUI/faceApp.ui", self)

        self.startButton.setCheckable(True)
        self.startButton.toggled.connect(self.start_webcam)

        self.detectButton.setCheckable(True)
        self.detectButton.toggled.connect(self.recognize_button)
        

        self.nameButton.setChecked(True)
        self.maskButton.setChecked(False)
        self.emotionButton.setChecked(False)
        self.genderButton.setChecked(False)
        self.ageButton.setChecked(False)
        self.blurButton.setChecked(False)
        self.faceLandmarksButton.setChecked(False)

        
        self.openVideoButton.clicked.connect(self.open_video)




        self.saveimageButton.clicked.connect(self.save_image)

        self.exitButton.clicked.connect(QApplication.instance().quit)

        self.webcam_Enabled = False
        self.recognize_enabled = False
        #self.record_Enabled = False

        self.ret = False
        self.source_camera = 0
        

        self.image = cv2.imread("GUI/Icon.png", 1)

        # Recognize
        self.userPath = 'media/faces'
        self.usersList = [i for i in os.listdir(self.userPath) if i.endswith('.jpg') or i.endswith('.png')]
        self.usersImgs, self.usersNames = self.list_userImg_userName(self.usersList, self.userPath)
        self.encodeListKnown = self.findEncodings()

        # Emotion
        self.face_exp_model = model_from_json(open('models/facial_expression_model_structure.json', 'r').read())
        self.face_exp_model.load_weights('models/facial_expression_model_weights.h5')
        self.emotions_labels = ('Wuetend', 'Angeekelt', 'Aengstlich', 'Gluecklich', 'Traurig', 'Ueberrascht', 'Neutral')

        # Gender
        self.gender_label_list = ["Maennlich", "Weiblich"]
        self.gender_protext = "models/gender_deploy.prototxt"
        self.gender_caffemodel = "models/gender_net.caffemodel"
        self.gender_cov_net = cv2.dnn.readNet(self.gender_caffemodel, self.gender_protext)
        # Age
        self.AGE_GENDER_MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        self.age_label_list = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60-100']
        self.age_protext = "models/age_deploy.prototxt"
        self.age_caffemodel = "models/age_net.caffemodel"
        self.age_cov_net = cv2.dnn.readNet(self.age_caffemodel, self.age_protext)

        # Mask detection
        self.prototxtPath = "models/deploy.prototxt"
        self.weightsPath = "models/res10_300x300_ssd_iter_140000.caffemodel"
        self.faceNet = cv2.dnn.readNet(self.prototxtPath, self.weightsPath)
        self.maskNet = load_model("models/mask_detector.model")

        # Face Landmarks
        # Load the detector
        self.detector = dlib.get_frontal_face_detector()

        # Load the predictor
        self.predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")


        # GUI
        self.adduserButton.clicked.connect(self.users_dialog)


        self.loadButton.clicked.connect(self.load_db)

        self.facecount = "0"

        self.path = 'media/faces'
        self.usersName = []



        
    
    def open_video(self):   
        self.source_camera = QFileDialog.getOpenFileName()
        self.source_camera = self.source_camera[0]
        self.capture = cv2.VideoCapture(self.source_camera)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)


    #### GUI ####
    def preview_image(self):
        
        names = list(set(self.usersName))

        if len(names) >= 1:
        
            name1 = names[0]
            path = self.path + '/' + str(name1) + '.jpg'
            thumbnail1 = QPixmap(path)
            thumbnail1 = thumbnail1.scaled(100, 90)
            self.userPhoto1.setPixmap(thumbnail1)
            self.userName1.setText(name1)

        if len(names) >= 2:
            name1 = names[0]
            path = self.path + '/' + str(name1) + '.jpg'
            thumbnail1 = QPixmap(path)
            thumbnail1 = thumbnail1.scaled(100, 90)
            self.userPhoto1.setPixmap(thumbnail1)

            name2 = names[1]
            path = self.path + '/' + str(name2) + '.jpg'
            thumbnail2 = QPixmap(path)
            thumbnail2 = thumbnail2.scaled(100, 90)
            self.userPhoto2.setPixmap(thumbnail2)
            self.userName2.setText(name2)


    

    def load_db(self):
        connection = sqlite3.connect('face.db')
        query = "SELECT mytime, mydate, name, mask FROM people ORDER BY person_id DESC"
        result = connection.execute(query)
        self.dbTableWidget.setRowCount(0)

        for row_number, row_data in enumerate(result):
            self.dbTableWidget.insertRow(row_number)
            for column_number, data in enumerate(row_data):
                self.dbTableWidget.setItem(row_number, column_number, QTableWidgetItem(str(data)))
        connection.close()


    def users_dialog(self):
        user = USER()
        user.exec_()



    def detect_face_mask(self, frame, faceNet, maskNet):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))
        faceNet.setInput(blob)
        detections = faceNet.forward()

        faces = []
        locs = []
        preds = []

        try:
            # loop over the detections
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    (startX, startY) = (max(0, startX), max(0, startY))
                    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                    face = frame[startY:endY, startX:endX]
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)
                    faces.append(face)
                    locs.append((startX, startY, endX, endY))

            # only make a predictions if at least one face was detected
            if len(faces) > 0:
                faces = np.array(faces, dtype="float32")
                preds = maskNet.predict(faces, batch_size=32)
            
            return (locs, preds)

        except:
            print('Mask Detection Error')

    def time(self):     # Get current time.
        return datetime.now().strftime("%d-%b-%Y:%I-%M-%S")

    def list_userImg_userName(self, usersList, userPath):
        usersImgs = []
        usersNames = []
        for user in usersList:
            userImg = cv2.imread('{}/{}'.format(userPath, user))
            usersImgs.append(userImg)
            userName = os.path.splitext(user)[0]
            usersNames.append(userName)
        return usersImgs, usersNames

    

    
    def predict_gender(self, face_roi):
        face_roi_blob = cv2.dnn.blobFromImage(face_roi, 1, (227, 227), self.AGE_GENDER_MODEL_MEAN_VALUES, swapRB=False)
        self.gender_cov_net.setInput(face_roi_blob)
        gender_predictions = self.gender_cov_net.forward()
        gender = self.gender_label_list[gender_predictions[0].argmax()]
        return gender

    def predict_age(self, face_roi):
        face_roi_blob = cv2.dnn.blobFromImage(face_roi, 1, (227, 227), self.AGE_GENDER_MODEL_MEAN_VALUES, swapRB=False)
        self.age_cov_net.setInput(face_roi_blob)
        age_predictions = self.age_cov_net.forward()
        age = self.age_label_list[age_predictions[0].argmax()]
        return age

    def predict_emotion(self, face_roi):
        # Gray and resize
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_roi = cv2.resize(face_roi, (48,48))
        # Array
        img_pixels = image.img_to_array(face_roi)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255
        # Predict
        exp_predictions = self.face_exp_model.predict(img_pixels)
        max_index = np.argmax(exp_predictions[0])
        emotion = self.emotions_labels[max_index]
        return emotion


    def recognize_button(self, status):
        if status:
            self.detectButton .setText('Erkennung beenden')
            self.recognize_enabled = True
        else:
            self.detectButton .setText('Gesicht erkennen')
            self.recognize_enabled = False

    def start_webcam(self, status):
        if status:
            camera = int(self.comboBox.currentText())
            self.capture = cv2.VideoCapture(camera)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(5)
            self.startButton .setText('Stop')
            self.webcam_Enabled = True
        else:
            self.startButton .setText('Webcam Starten')
            self.webcam_Enabled = False
            self.timer.stop()
            self.ret = False
            self.capture.release()
            cv2.destroyAllWindows()


    def known_faces(self, encodeFace):
    	all_matches = face_recognition.compare_faces(self.encodeListKnown, encodeFace)
    	name = 'UNBEKANNT'
    	if True in all_matches:
    		first_match_index = all_matches.index(True)
    		name = self.usersNames[first_match_index].upper()
    	return name

    def findEncodings(self):
    	encodeList = []
    	for img in self.usersImgs:
    		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    		encode = face_recognition.face_encodings(img)[0]
    		encodeList.append(encode)
    	return encodeList

    
    def face_blur(self, face_roi):
        face_roi_blur = cv2.GaussianBlur(face_roi, (99,99), 30)
        return face_roi_blur


    def draw_face_landmarks(self,frame):     
        # Convert image into grayscale
        gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
        # Use detector to find landmarks
        faces = self.detector(gray)

        for face in faces:
            # Create landmark object
            landmarks = self.predictor(image=gray, box=face)

            # Loop through all the points
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y

                # Draw a circle
                cv2.circle(img=frame, center=(x, y), radius=2, color = (255,255,0), thickness=-1)
        #return frame


    def draw_rectangle(self, frame, top, right, bottom, left, color, text):
        cv2.rectangle(frame, (left,top), (right,bottom), color, 2)
        cv2.rectangle(frame, (left,bottom-35), (right,bottom), color,cv2.FILLED)
        cv2.putText(frame,text.upper(), (left+6,bottom-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)

    
    ##### DATABASE #####
    def data_entry(self, connection, cursor,  name, mask, date, time):
        cursor.execute("INSERT INTO people VALUES (NULL, ?, ?, ?, ?)", (name, mask, date, time))
        connection.commit()
        print(name, mask, date, time)


    def attendance(self, name, mask):
        now = datetime.now()
        date = str(now.strftime("%d/%m/%Y"))
        time = str(now.strftime("%H:%M:%S"))
        hour = time.split(":")[0]

        try:
            connection = sqlite3.connect("face.db")
            cursor = connection.cursor()
            cursor.execute("SELECT name, mydate, mytime, mask FROM people  WHERE name=? ORDER BY person_id DESC LIMIT 1", (name,))
            row = cursor.fetchall()
            
            if row == []:
                self.data_entry(connection, cursor,  name, mask, date, time)
            else:   

                dbName, dbDate, dbTime, dbMask = row[0]
                dbHour = dbTime.split(":")[0]

                if dbName==name and dbDate==date and dbHour==hour:
                    pass
                else:
                    if name != 'UNBEKANNT':
                        self.data_entry(connection, cursor,  name, mask, date, time)

            cursor.close()
            connection.close()
            
            #print("Success DB")
        except:
            print("Error DB")


    ##### Face Recognition Main Function #####
    def recognize(self, frame):
        frameSmall = cv2.resize(frame,(0,0),None,0.25,0.25)
        frameSmall = cv2.cvtColor(frameSmall, cv2.COLOR_BGR2RGB)

        facesCurrentFrame = face_recognition.face_locations(frameSmall,number_of_times_to_upsample=1,model='hog')
        encodesCurrentFrame = face_recognition.face_encodings(frameSmall,facesCurrentFrame)
        
        self.facecount = str(len(facesCurrentFrame))
        self.countLabel.setText(self.facecount)
        

        if len(facesCurrentFrame) >= 1:
            self.preview_image()

        for encodeFace,faceLocation in zip(encodesCurrentFrame, facesCurrentFrame):
            top, right, bottom, left = faceLocation
            top, right, bottom, left = top*4, right*4, bottom*4, left*4
            face_roi = frame[top:bottom,left:right]
            

            name = self.known_faces(encodeFace)
            
            if name != 'UNBEKANNT':
            	self.usersName.append(name)             

            if self.maskButton.isChecked():
                locs, preds = self.detect_face_mask(frame, self.faceNet, self.maskNet)
                
                for (box, pred) in zip(locs, preds):
                    #left, top, right, botttom = box
                    (mask, withoutMask) = pred
                    label = "Maske" if mask > withoutMask else "Keine Maske"
                    color = (0, 255, 0) if label == "Maske" else (0, 0, 255)   
                    self.draw_rectangle(frame, top, right, bottom, left, color, label)
                    self.attendance(name, label)
            
            elif self.emotionButton.isChecked():
                emotion = self.predict_emotion(face_roi)
                color = (0,0,255)
                self.draw_rectangle(frame, top, right, bottom, left, color, emotion)

            elif self.genderButton.isChecked():
                gender = self.predict_gender(face_roi)
                color = (0,0,255)
                self.draw_rectangle(frame, top, right, bottom, left, color, gender)

            elif self.ageButton.isChecked():
                age = self.predict_age(face_roi)
                color = (0,0,255)
                self.draw_rectangle(frame, top, right, bottom, left, color, age)
            
            elif self.blurButton.isChecked():
                face_roi_blur = cv2.GaussianBlur(face_roi, (99,99), 30)
                frame[top:bottom, left:right] = face_roi_blur

            elif self.faceLandmarksButton.isChecked():
                self.draw_face_landmarks(frame)

            else:
                color = (0,0,255)
                self.draw_rectangle(frame, top, right, bottom, left, color, name)

        return frame



    ##### Video Frame #####
    def update_frame(self):
        self.ret, self.image = self.capture.read()
        if self.ret:
            self.image = cv2.flip(self.image, 1)
            detected_image = None

            if (self.recognize_enabled):
                # Recognize
                detected_image = self.recognize(self.image)
                self.displayImage(self.image, 1)

            else:
                self.displayImage(detected_image, 1)


    def displayImage(self, img, window=1):
    	pixImage = self.pix_image()
    	if window == 1:
    		self.imageLabel.setPixmap(QPixmap.fromImage(pixImage))
    		self.imageLabel.setScaledContents(True)


    # Converting image from OpenCv to PyQT compatible image.
    def pix_image(self):
        qformat = QImage.Format_RGB888  # only RGB Image
        if len(self.image.shape) == 3:
            r, c, ch = self.image.shape
        else:
            r, c = self.image.shape
            qformat = QImage.Format_Indexed8
        pixImage = QImage(self.image, c, r, self.image.strides[0], qformat)
        return pixImage.rgbSwapped()


    ##### SAVE DATA #####
    # Save image captured using the save button.
    def save_image(self):
        location = QFileDialog.getSaveFileName(self, 'Save File')
        location = location[0]
        #location =  location[0] 
        file_type = ".jpg"
        file_name = location+file_type 
        try:
            cv2.imwrite(file_name, self.image)
            QMessageBox().about(self, "Image Saved", "Saved successfully at "+file_name)
        except:
            print('Error Save Images')



##### MAIN FUNCTION #####
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    window = FaceApp()         # Running application loop.
    window.setWindowTitle("Face App")
    window.show()

    sys.exit(app.exec_())       #  Exit applicatio




