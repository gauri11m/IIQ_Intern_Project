import mrcnn.model as modellib
from mrcnn.config import Config
from visualise import random_colors, get_mask_contours, draw_mask
import smtplib
import datetime
import time
import mysql.connector

mydb=mysql.connector.connect(host="localhost",user="root",password="intersectiq", database="aditi")
mycursor=mydb.cursor()

import cv2


class CustomConfig(Config):
    def __init__(self, num_classes):
        super().__init__()
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "object"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4

    # Number of classes
    NUM_CLASSES = 1 + 2

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    # TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 500

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

    DETECTION_MIN_CONFIDENCE = 0.9

class InferenceConfig(CustomConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1



def load_inference_model(num_classes, model_path):
    inference_config = InferenceConfig(num_classes)

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=model_path)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
    #model_path = model.find_last()

    # Load trained weights
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    return model, inference_config

test_model, inference_config = load_inference_model(1, r"mask_rcnn_object_0019.h5")
print('Hi')
# Load Image

class image:
    def seg(self,img,id_list,id):
        img = cv2.resize(img, (1280, 720), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
        print('Hello')

        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect results
        r = test_model.detect([image])[0]
        colors = [(255, 0, 255), (255, 255, 0), (0, 255, 255), (0, 0, 0), (185, 0, 0), (185, 255, 0), (0, 255, 185)]
        # Get Coordinates and show it on the image
        # Get Coordinates and show it on the image
        object_count = len(r["class_ids"])
        ar = []
        max_area = 0
        contourss = []
        area = 0
        for i in range(object_count):
            # 1. Mask
            mask = r["masks"][:, :, i]
            contours = get_mask_contours(mask)
            # print(contours)

            area1 = 0

            # print(contours)
            for cnt in contours:
                cv2.polylines(img, [cnt], True, colors[i], 2)
                # img = draw_mask(img, [cnt], colors[i])

                area1 = area1 + cv2.contourArea(cnt)
                # print(cnt)
                # break

            area = area + area1
            if area1 > max_area:
                max_area = area1
                # contourss = cnt
            # ar.append(area)
            # break
        if (max_area > 0):
            cleanliness = 100 - ((area - max_area) / max_area) * 100
        else:
            cleanliness = 100

        # creates SMTP session
        s = smtplib.SMTP('smtp.gmail.com', 587)

        # start TLS for security
        s.starttls()

        # Authentication
        s.login("s76149951@gmail.com", "intersectiq")
        # s.login("noreply@intersectiq.com", "Innovate@2020")

        # message to be sent
        # message = "Message_you_need_to_send"

        message = """From: From Person <s76149951@gmail.com>
                            Subject: Anomaly detected Unclean Tanker .format(SUBJECT, TEXT)

                            An anomaly has been detected..
                            """

        # sending the mail
        sqlquery = "Insert into trucks(id,time,status) values(%s,%s,%s)"
        millisec = int(round(time.time() * 1000))

        k = 0
        if (cleanliness < 70):
            k = 1
            current_time = datetime.datetime.now()
            millisec = int(round(time.time() * 1000))
            cv2.putText(img, f'{cleanliness}', (100, 100),
                        cv2.FONT_HERSHEY_PLAIN, 2.0,
                        (0, 255, 255), 2)
            cv2.imwrite(r"C:\Users\Hp\PycharmProjects\BPCL-final\Mask_RCNN\anomalies\pic" + ' ' + str(
                millisec) + '.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 30])
            # cv2.imwrite(r'C:\Users\Hp\PycharmProjects\BPCL-final\Mask_RCNN\anomilies\pic1.jpeg',img)
            s.sendmail("s76149951@gmail.com", "aditi29.2015@gmail.com", message)
        if(k==1):
            info = [(id, millisec,'dirty')]
        else:
            info = [(id, millisec, 'clean')]

        mycursor.executemany(sqlquery, info)
        mydb.commit()

        # terminating the session
        s.quit()

        print(id_list)

