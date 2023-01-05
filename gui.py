import tkinter as tk
import customtkinter as ctk 

import torch
import numpy as np

import cv2
from PIL import Image, ImageTk
import vlc 
import random
import time

app = tk.Tk()
app.geometry("600x600")
app.title("Sleepi Boi")
app.iconbitmap("icon.ico")
ctk.set_appearance_mode("dark")

vidFrame = tk.Frame(height=480, width=600)
vidFrame.pack()
vid = ctk.CTkLabel(vidFrame)
vid.pack()

counter = 0 
counterLabel = ctk.CTkLabel(master = app, text=counter,  height=40, width=120, text_color="black")
counterLabel.pack(pady=10)

def reset_counter(): 
    global counter
    counter = 0 
resetButton = ctk.CTkButton(master = app, text="Reset Counter", command=reset_counter, height=40, width=120) 
resetButton.pack()

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp/weights/last.pt', force_reload=True)
cap = cv2.VideoCapture(0)
def detect(): 
    global counter
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    results = model(frame) 
    img = np.squeeze(results.render())

    if len(results.xywh[0]) > 0: 
        dconf = results.xywh[0][0][4]
        dclass = results.xywh[0][0][5]
        print(results.xywh[0])
        if dconf.item() > 0.85 and dclass.item() == 16.0:
            # filechoice = random.choice([1,2,3])
            p = vlc.MediaPlayer(f"file:///audio.wav")
            p.play()
            time.sleep(2)
            counter += 1 

    imgarr = Image.fromarray(img)
    # imgtk = ImageTk.PhotoImage(imgarr) 
    imgtk = ctk.CTkImage(imgarr, size=(600, 480))
    vid.imgtk = imgtk
    vid.configure(image=imgtk)
    vid.after(10, detect) 
    counterLabel.configure(text=counter)  


detect()
app.mainloop()