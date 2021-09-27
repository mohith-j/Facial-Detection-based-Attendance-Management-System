# -*- coding: utf-8 -*-
"""
Created on Fri 13 Mar 2020
"""
import tkinter as tk #interface
import cv2,os #opencv,os things
import csv #for excel 
import numpy as np 
import pandas as pd
import datetime #time
import time #time
#from tkinter import Message ,Text
##import tkinter.ttk as ttk
#import tkinter.font as font
#import shutil

window = tk.Tk()
window.title("Face_Recognizer")
dialog_title = 'QUIT'
dialog_text = 'Are you sure?'
window.geometry('1440x1440')
window.configure(background='SteelBlue3')
#window.attributes('-fullscreen', True)
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

message = tk.Label(window, text="Authentication System using Facial Recognition" ,bg="SteelBlue4"  ,fg="white"  ,width=50  ,height=3,font=('Segoe UI Light', 30, 'underline')) 
message.place(x=200, y=20)

lbl = tk.Label(window, text="Enter ID : ",width=20, height=2, fg="azure", bg="SteelBlue4", font=('Segoe UI Semibold', 15, 'bold')) 
lbl.place(x=400, y=200)

txt = tk.Entry(window,width=20,bg="SteelBlue4", fg="azure", font=('Segoe UI', 15))
txt.place(x=700, y=215)

lbl2 = tk.Label(window, text="Enter Name : ", width=20, fg="azure", bg="SteelBlue4", height=2, font=('Segoe UI Semibold', 15, 'bold')) 
lbl2.place(x=400, y=300)

txt2 = tk.Entry(window,width=20  ,bg="SteelBlue4", fg="azure", font=('Segoe UI', 15))
txt2.place(x=700, y=315)

lbl3 = tk.Label(window, text="Status : ", width=20, fg="azure", bg="SteelBlue4", height=2, font=('Segoe UI Semibold', 15, 'bold')) 
lbl3.place(x=400, y=400)

message = tk.Label(window, text="", bg="SteelBlue4", fg="azure", width=30, height=2, activebackground = "SteelBlue3" , font=('Segoe UI Light', 15, 'bold')) 
message.place(x=700, y=400)

lbl3 = tk.Label(window, text="Entry Log : ", width=20, fg="azure", bg="SteelBlue4", height=2, font=('Segoe UI Semibold', 15, 'bold')) 
lbl3.place(x=400, y=650)


message2 = tk.Label(window, text="" , fg="azure", bg="SteelBlue4", activeforeground = "green",width=30  ,height=2  ,font=('Segoe UI', 15, ' bold ')) 
message2.place(x=700, y=650)
 
def clear(): #GUI clear
    txt.delete(0, 'end')    
    res = ""
    message.configure(text= res)

def clear2(): #GUI clear
    txt2.delete(0, 'end')    
    res = ""
    message.configure(text= res)    
    
def is_number(s): 
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False
 
def TakeImages():         
    Id=(txt.get())
    name=(txt2.get())
    if(is_number(Id) and name.isalpha()):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector=cv2.CascadeClassifier(harcascadePath)
        sampleNum=0
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #bounding box       
                #incrementing sample number 
                sampleNum=sampleNum+1
                #saving the captured face in the dataset folder TrainingImage
                cv2.imwrite("TrainingImage\ "+name +"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                #display the frame
                cv2.imshow('frame',img)
            #wait for 100 miliseconds 
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is more  than 50
            elif sampleNum>50:
                break
        cam.release()
        cv2.destroyAllWindows() 

        # file handling portion
        res = "Images Saved for ID : " + Id +" Name : "+ name
        row = [Id , name]
        with open('UserDetails\\UserDetails.csv','a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text= res)
    else:
        if(is_number(Id)):
            res = "Enter Alphabetical Name"
            message.configure(text= res)
        if(name.isalpha()):
            res = "Enter Numeric Id"
            message.configure(text= res)
    
def TrainImages():
    recognizer = cv2.face_LBPHFaceRecognizer.create()#recognizer = cv2.face.LBPHFaceRecognizer_create()#$cv2.createLBPHFaceRecognizer()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector =cv2.CascadeClassifier(harcascadePath)
    faces,Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel\Trainner.yml")
    res = "Image Trained"#+",".join(str(f) for f in Id)
    message.configure(text= res)

def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    #print(imagePaths)
    
    #create empth face list
    faces=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id)        
    return faces,Ids

def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()#cv2.createLBPHFaceRecognizer()
    recognizer.read("TrainingImageLabel\Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);    
    df=pd.read_csv("UserDetails\\UserDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX        
    col_names =  ['Id','Name','Date','Time']
    attendance = pd.DataFrame(columns = col_names)    
    while True:
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)    
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])                                   
            if(conf < 50):
                ts = time.time()      
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa=df.loc[df['Id'] == Id]['Name'].values
                tt=str(Id)+"-"+aa
                attendance.loc[len(attendance)] = [Id,aa,date,timeStamp]
                
            else:
                Id='Unknown'                
                tt=str(Id)  
            if(conf > 75):
                noOfFile=len(os.listdir("ImagesUnknown"))+1
                cv2.imwrite("ImagesUnknown\Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])            
            cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)        
        attendance=attendance.drop_duplicates(subset=['Id'],keep='first')    
        cv2.imshow('im',im) 
        if (cv2.waitKey(1)==ord('q')):
            break
    ts = time.time()      
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    fileName="EntryLog\\EntryLog_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    attendance.to_csv(fileName,index=False)
    cam.release()
    cv2.destroyAllWindows()
    #print(attendance)
    res=attendance
    message2.configure(text= res)

  
clearButton = tk.Button(window, text="Clear", command=clear, fg="azure", bg="SteelBlue4", width=20, height=2, activebackground = "Red" ,font=('Segoe UI Semibold', 15, 'bold'))
clearButton.place(x=950, y=200)
clearButton2 = tk.Button(window, text="Clear", command=clear2, fg="azure", bg="SteelBlue4", width=20  ,height=2, activebackground = "Red" ,font=('Segoe UI Semibold', 15,'bold'))
clearButton2.place(x=950, y=300)    
takeImg = tk.Button(window, text="Take Images", command=TakeImages, fg="SteelBlue4", bg="SlateGray1", width=20, height=3, activebackground = "Red" ,font=('Segoe UI Semibold', 15, ' bold '))
takeImg.place(x=200, y=500)
trainImg = tk.Button(window, text="Train Images", command=TrainImages, fg="SteelBlue4", bg="SlateGray1", width=20, height=3, activebackground = "Red" ,font=('Segoe UI Semibold', 15, ' bold '))
trainImg.place(x=500, y=500)
trackImg = tk.Button(window, text="User Login", command=TrackImages, fg="SteelBlue4", bg="SlateGray1", width=20, height=3, activebackground = "Red" ,font=('Segoe UI Semibold', 15, ' bold '))
trackImg.place(x=800, y=500)
quitWindow = tk.Button(window, text="Quit", command=window.destroy, fg="SteelBlue4", bg="SlateGray1", width=20, height=3, activebackground = "Red" ,font=('Segoe UI Semibold', 15, ' bold '))
quitWindow.place(x=1100, y=500)
copyWrite = tk.Text(window, background=window.cget("background"), borderwidth=0,font=('Segoe UI Semibold', 30, 'underline'))
copyWrite.tag_configure("superscript", offset=10)
#copyWrite.insert("insert", "Developed by 17BCE2116")
copyWrite.configure(state="disabled",fg="red")
copyWrite.pack(side="left")
copyWrite.place(x=800, y=750)
 
window.mainloop()