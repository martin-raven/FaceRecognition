import face_recognition
from os import listdir
import os
import cv2
import pickle
import face_recognition
def getLabel(result,label):
    j=0
    for i in result:
        if(i):
            print("Image has "+label[j]+"in it .")
        j+=1 
TrainingData=[]
labels=[]
TestData=[]
if(not(os.path.isfile("TrainingData.pkl"))):
    trainedfile = open('TrainingData.pkl', 'wb')
    Labelfile = open('LabelData.pkl', 'wb')
    TraningFiles=listdir("TrainingData")
    for Image in TraningFiles:
        IMAGE=face_recognition.load_image_file("TrainingData"+"/"+Image)
        try:
            TrainingData.append(face_recognition.face_encodings(IMAGE)[0])
            labels.append(Image)
        except:
            print("I wasn't able to locate any faces in at least one of the images. Check the image files: "+Image)
    pickle.dump(TrainingData, trainedfile)
    pickle.dump(labels,Labelfile)
else:
    trainedfile = open('TrainingData.pkl', 'rb')
    Labelfile = open('LabelData.pkl', 'rb')
    TrainingData=pickle.load(trainedfile);
    labels=pickle.load(Labelfile)
print(TrainingData)
TestFiles=listdir("TestData")
for Image in TestFiles:
    IMAGE=face_recognition.load_image_file("TestData"+"/"+Image)
    try:
        TestData.append(face_recognition.face_encodings(IMAGE)[0])
    except IndexError:
        print("I wasn't able to locate any faces in at least one of the images. Check the image files. "+Image)
for test in TestData:
    results = face_recognition.compare_faces(TrainingData, test)
    getLabel(results,labels)
