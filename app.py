#Libararies Needed
import tkinter as tk
import cv2,os
import csv
import numpy as np
import pandas as pd
import datetime
import time
from PIL import Image

# USER INTERFACE PORTION STARTS

"""
UI is made using Tkinter
UI is made up of widgets

We create a window for the app on which widgets will be added
We set the size and the name of the app and the background color.
"""
window = tk.Tk()
window.title("Face_Recognizer")
window.geometry('1440x1440')
window.configure(background='SteelBlue3')
#The below line is used if we want the app to run in fullscreen size
#window.attributes('-fullscreen', True)

##weight=1 means we configure window so that widgets will expand or shrink to fill the window
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)


"""
#Types of Widgets used:
#Label - Text that is displayed
#Entry - Entry field (Text input field)
#Button - Button

Widget parameters include the text to display, fg and bg color, font and font style, height and width
"""


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
 
# USER INTERFACE PORTION ENDS

#customizable function for pattern matching
#right now just verifies if input is a float number
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        #if it didn't return True we will check it later 
        pass
 
    
#Captures image of new user and adds user to database
#FACE DETECTION PORTION STARTS  
def TakeImages():        
    Id=(txt.get())
    name=(txt2.get())
    
    #check NAME is alphabet string, ID is numeric
    if(is_number(Id) and name.isalpha()):
        
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector=cv2.CascadeClassifier(harcascadePath)
        
        #counts number of sample images stored in folder
        sampleNum=0
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            """detectMultiScale(image, scaleFactor, minNeighbors=3) this function detects face
            Reducing minNeighbours increases false positive face detections (i.e.incorrect detection of a face when there is no face)
            """
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                #Creating the bounding box around face
                #cv2.rectangle(image, start_point, end_point, color, thickness) 
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)        
                #incrementing sample number 
                sampleNum=sampleNum+1
                #saving the captured face in the dataset folder TrainingImage
                cv2.imwrite("TrainingImage\ "+name +"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                #display the frame around the face
                cv2.imshow('frame',img)
            #wait for 100 miliseconds or if user force quits by pressing 'q' on keyboard
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # or just stop when at least 50 sample images are taken
            elif sampleNum>50:
                break
        cam.release()
        cv2.destroyAllWindows() 

        
        # FILE HANDLING PORTION
        res = "Images Saved for ID : " + Id +" Name : "+ name
        # These details will be stored in UserDetails file
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


#Face Recognition training portion starts    
def TrainImages():
    """
    We will use as a recognizer, the LBPH Face Recognizer available OpenCV
    We do this in the following line:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    The function “getImagesAndLabels (path)”, will take all photos on directory: “dataset/”, 
    and returns 2 arrays: “Ids” and “faces”. 
    With those arrays as input, we will “train our recognizer”:
        recognizer.train(faces, ids)
    As a result, a file named “trainner.yml” will be saved in the trainer directory that was previously created by us.
    """
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    faces,Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel\Trainner.yml")
    
    #Display following message in status bar
    res = "Image Trained"
    message.configure(text= res)

#user defined function to receive arrays of faces and IDs
def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    #create empty face list
    faces=[]
    #create empty ID list
    Ids=[]
    
    #go through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the greyscale image and converting it to L mode (luminance mode)
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        # extract the face from the training image sample
        faces.append(imageNp)
        
        #getting the Id from the filename ("imagePath/Name.ID.1.jpg")
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        Ids.append(Id)        
    return faces,Ids


#FACE RECOGNITION TESTING PORTION STARTS    
# Now the trained model will be tested on image of user who is trying to login
# This portion again includes Face Detection as we need to first detect the face from the live web cam input
    
def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel\Trainner.yml")
    
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);    
    
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX        
    #These are the values that will be stored in the Entry Log
    #Right now we store it in a Pandas dataframe, which we later conver to csv file
    col_names =  ['Id','Name','Date','Time']
    attendance = pd.DataFrame(columns = col_names)    
    """
    We also read the User Details csv file into a Pandas dataframe (we later go through it and see if there is a match)
    """
    df=pd.read_csv("UserDetails\\UserDetails.csv")
    
    
    while True:
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)    
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            
            """
            Now the trained recognizer model has to predict whether the live input
            face is of a recognized person or unknown person.
            
            conf = confidence
        
            If the confidence is higher, then it means that the pictures are less similar.
            In other words the lower, the better
            """
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])           
            #                        
            if(conf < 50):
                #timestamp - find current system time and convert to string format
                ts = time.time()      
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                
                #find match ID with ID in UserDetails and find corresponding name
                userName=df.loc[df['Id'] == Id]['Name'].values
                tt=str(Id)+"-"+userName
                attendance.loc[len(attendance)] = [Id,userName,date,timeStamp]
                
            else:
                Id='Unknown'                
                tt=str(Id)
                
            #Take images if % match of face is less than 20% (Intruder Detection)
            if(conf > 80):
                noOfFile=len(os.listdir("ImagesUnknown"))+1
                cv2.imwrite("ImagesUnknown\Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])            
            
            #Write text on the live image
            #putText() shows the name and ID of the person who is in the image
            #shows Unknown if there was no match
            cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)        
        
        """
        While the for loop is running attendance of same person keeps being added to Pandas dataframe
        Obviously we won't keep duplicate records in the EntryLog (Data Redundancy)
        So we delete the duplicates from the dataframe and only keep the first recorded entry of that person
        """
        attendance=attendance.drop_duplicates(subset=['Id'],keep='first')    
        
        
        cv2.imshow('im',im) 
        if (cv2.waitKey(1)==ord('q')):
            break
    
    #FILE HANDLING PORTION - Dataframe is converted to csv file here
    ts = time.time()      
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    fileName="EntryLog\\EntryLog_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    attendance.to_csv(fileName,index=False)
    
    
    #release web cam for other apps
    cam.release()
    #destroy the web cam pop up window
    cv2.destroyAllWindows()
    #display in app that attendance recorded
    res=attendance
    message2.configure(text= res)


# USER INTERFACE PORTION STARTS
#All the 6 buttons are described here - 

#User defined functions for later use that are connected to CLEAR buttons
#Clears text field input (ID)
def clear():
    txt.delete(0, 'end')    
    res = ""
    message.configure(text= res)

#Clears text field input (NAME)
def clear2():
    txt2.delete(0, 'end')    
    res = ""
    message.configure(text= res)   

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

#runs till app is exited OR someone presses Quit button
window.mainloop()