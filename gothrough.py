import numpy as np
import cv2
import time
from PIL import Image
from edgetpu.detection.engine import DetectionEngine
import sys 

###########################################neopixel init
print("welcome!!!!")

import board
import neopixel

#pixel_pin = board.D18
#num_pixels = 144
#ORDER = neopixel.GRB

pixels = neopixel.NeoPixel(
    board.D18, 144, brightness=0.2, auto_write=False, pixel_order=neopixel.GRB
)

########################################### mqtt init

import paho.mqtt.client as mqtt

topic = "pi_to_lolin"
topic2 = "lolin_to_pi"

def on_connect(client, userdata, flags, rc):
    print(" Connected with result code "+str(rc))
    client.subscribe(topic2)


def on_message(client, userdata, msg):
        str2 = msg.payload.decode()
        
        print(str2)



client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect("192.168.2.1", 1883, 60)
client.loop_start()
#############################################



pixels.fill((0, 255, 0))
pixels.show()



#model = "models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
label_path = "/home/pi/examples/models/coco_labels.txt"

# creating DetectionEngine with model
engine = DetectionEngine("/home/pi/examples/models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite")

labels = {}
box_color = [0, 255, 0]
prevTime = 0

# reading label file
with open(label_path, 'r') as f:
    lines = f.readlines()
    for line in lines:    # ex) '87 teddy bear'
        id, name = line.strip().split(maxsplit=1)   # ex) '87', 'teddy bear'
        labels[int(id)] = name
print(f"Trained object({len(labels)}):\n{labels.values()}")
print("Quit to ESC.")

cap = cv2.VideoCapture(-1)

temp = True
check = True
check2 = True


while True:
    ret, frame = cap.read()
                    
    currTime = time.time()
    if(currTime - prevTime < 1):
        temp = False
    else:
        temp = True
    if not ret:
        print("cannot read frame.")
        break
    img = frame[:, :, ::-1].copy()  # BGR to RGB
    img = Image.fromarray(img)  # NumPy ndarray to PIL.Image

    candidates = engine.detect_with_image(img, threshold=0.3, top_k=3, keep_aspect_ratio=True, relative_coord=False, )

    if candidates:
        for obj in candidates:
            if obj.label_id == 0:
                # the same color for the s$
                #client.publish(topic, "Detected!!!")
                
                if(check):
                    pixels.fill((255, 0, 0))
                    pixels.show()
                    client.publish(topic, "F")
                    check = False

                prevTime = currTime
                

            # drawing bounding-box
                box_left, box_top, box_right, box_bottom = tuple(map(int, obj.bounding_box.ravel()))
                cv2.rectangle(frame, (box_left, box_top), (box_right, box_bottom), box_color, 2)

            # drawing label box
                accuracy = int(obj.score * 100)
                label_text = labels[obj.label_id] + " (" + str(accuracy) + "%)" 
                (txt_w, txt_h), base = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_PLAIN, 2, 3)
                cv2.rectangle(frame, (box_left - 1, box_top - txt_h), (box_left + txt_w, box_top + txt_h), box_color, -1)
                cv2.putText(frame, label_text, (box_left, box_top+base), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255,255), 2)
            else:
                if temp and check == False:
                    pixels.fill((0, 255, 0))
                    pixels.show()
                    client.publish(topic, "P")
                    check = True
    else:
        if temp and check == False:
            pixels.fill((0, 255, 0))
            pixels.show()
            client.publish(topic, "P")
            check = True
    # calculating and drawing fps            
    #currTime = time.time()
    #fps = 1/ (currTime -  prevTime)
    #prevTime = currTime
    #cv2.putText(frame, "fps:%.1f"%fps, (10,30), cv2.FONT_HERSHEY_PLAIN, 2, (0,255, 0), 2)
    #cv2.putText(frame, "", (10,30), cv2.FONT_HERSHEY_PLAIN, 2, (0,255, 0), 2)
    #reimg = cv2.resize(frame, None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
    #h, w = frame.shape[:2]
    #w = 640, h = 480
    #print(h, end="hhhhhhhhhhhhhhh")
    #print(w, end="wwwwwwwwwwwwwww")
    #cv2.imshow('Object Detecting', reimg2)

    cv2.imshow('Object Detecting', frame)
    if cv2.waitKey(1)&0xFF == 27:
        client.disconnect()
        pixels.fill((0, 0, 0))
        pixels.show()
        break  
cap.release()
