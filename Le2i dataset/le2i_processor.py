import os
import cv2
import json
import random
import numpy as np
import mediapipe as mp

import sys
sys.path.append("../")
from landmark_extractor import LandmarkExtractor

'''
Le2i Dataset structure
citation: https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia
6 folders, each one  with a falling scenario
Label all activities as '1' or '0' after checking

Read from each sigle folder, there is another nested folder with the 'Videos' folder get landmarks and wrap into temporal sequences of 3 seconds
'''


def process_dataset():
    landmark_extractor = LandmarkExtractor()

    previous_landmarks = None

    sequence_length = 90
    frames_counter = 0

    data = []
    dataset_path = "/home/autonomouslab/Downloads/Le2i/Office"

    for video_sequence in sorted(os.listdir(os.path.join(dataset_path, "Office"))):
        print(f"Processing sequence {video_sequence}")

        previous_landmarks = None
        queue = []
        bl = []
        labels = []
        images = []

        video_cap = cv2.VideoCapture(os.path.join(dataset_path, "Office", video_sequence))
        ret, image = video_cap.read()

        while ret:

            body_landmarks, body_ar = landmark_extractor.get_body_landmarks(image)
            if body_landmarks:
                body_landmarks = fix_wrist_landmarks(body_landmarks)

                check, keys = check_body_landmarks(body_landmarks)
                if not check:
                    print(f"{keys} landmarks are out of image!")
                previous_landmarks = body_landmarks

            else:
                body_landmarks = previous_landmarks
            
            if body_landmarks is not None:
                feature_vector = vectorize_landmarks(body_landmarks, body_ar)
                queue.append(feature_vector)
                bl.append(body_landmarks)
                labels.append(1)
                images.append(image)

            ret, image = video_cap.read()

        skip = int(15)

        if len(queue) > 200:
            skip = int(30)

        if len(queue) > 1000:
            skip = int(60)

        for i in range(0, len(queue) - sequence_length + 1, skip):
            print("New sequence...")
            print(f"Going from {i} to {i+sequence_length}")
            row = []

            for j in range(sequence_length):
                row.append(queue[i + j])

                image = images[i+j]

                body_landmarks = bl[i + j]
                normalized_bl = landmark_extractor.de_normalize_body_landmarks(body_landmarks)
                out_img = landmark_extractor.draw_selected_landmarks(normalized_bl, image)
                cv2.imshow('', out_img)
                cv2.waitKey(20)
            
            
            ques = input("Save this sequence? \n")
            if ques == "0":
                row.append(0)
                data.append(row)

                print("Saved...")
                print(f"Dataset has now {len(data)} samples")
            
            elif ques == "1":
                row.append(1)
                data.append(row)

                print("Saved...")
                print(f"Dataset has now {len(data)} samples")
            
            elif ques == "s":
                break

            else:
                print("Discarded")

    dump_json(data)       



def dump_json(data):
    out_path = "/home/autonomouslab/Desktop/assistive_robodog/Perception/Dataset/Data_Files/le2i_office.json"

    with open(out_path, 'w') as json_file:
        json_file.write('[\n')
        for row_idx, row in enumerate(data):
            json_file.write('  [\n')
            for vector_idx, vector in enumerate(row):
                vector_str = '    ' + json.dumps(vector)
                if vector_idx < len(row) - 1:
                    json_file.write(f'{vector_str},\n')
                else:
                    json_file.write(f'{vector_str}\n')
            if row_idx < len(data) - 1:
                json_file.write('  ],\n')
            else:
                json_file.write('  ]\n')
        json_file.write(']\n')



def vectorize_landmarks(landmarks, body_ar):
    input_vector = []
    for key in landmarks.keys():
        input_vector.append(float(landmarks[key][0]))
        input_vector.append(float(landmarks[key][1]))

    input_vector.append(body_ar)
    return input_vector
    

def fix_wrist_landmarks(landmarks):
    '''
    Hand landmark detector does not capture wrist landmarks, in this case the best thing is to assume person falling with
    straight arms pointing the floor, so wrist landmarks can be similar to hip landmarks
    '''
    rand_x = random.uniform(0.001, 0.01)
    rand_y = random.uniform(0.001, 0.01)

    if landmarks['left_wrist'] == [0, 0]:
        landmarks['left_wrist'][0] = landmarks['left_hip'][0] + rand_x
        landmarks['left_wrist'][1] = landmarks['left_hip'][1] + rand_y
    if landmarks['right_wrist'] == [0, 0]:
        landmarks['right_wrist'][0] = landmarks['right_hip'][0] + rand_x
        landmarks['right_wrist'][1] = landmarks['right_hip'][1] + rand_y
    return landmarks


def check_body_landmarks(coords):
    for key in coords.keys():
        for landmark in coords[key]:
            if coords[key][0] is None:
                return False, key
            elif coords[key][0] == float(0) and coords[key][1] == float(0):
                return False, key
    return True, "None"


if __name__ == "__main__":
    process_dataset()