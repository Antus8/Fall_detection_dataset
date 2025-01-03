import os
import cv2
import yaml
import json
import random
import numpy as np
import mediapipe as mp

import sys
sys.path.append("../Dataset Tools")
from landmark_extractor import LandmarkExtractor

'''
MONTREAL Dataset structure
citation: http://www.iro.umontreal.ca/~labimage/Dataset/
24 folders, one for each scenario

Each folder has the same scene recorded from different point of views
'''


def main():

    config_params = load_config("config.yaml")

    landmark_extractor = LandmarkExtractor()

    previous_landmarks = None

    sequence_length = config_params['dataset_processor_params']['cam_fps'] * config_params['dataset_processor_params']['sequence_length']

    data = []
    dataset_path = config_params['dataset_processor_params']['dataset_path']

    for folder in sorted(os.listdir(dataset_path)):
        # Process each experiment (in a separate folder)
        queue = []
        bl = []
        labels = []
        images = []

        for video_sequence in sorted(os.listdir(os.path.join(dataset_path, folder))):
            print(f"Processing sequence {os.path.join(folder, video_sequence)}")
            video_cap = cv2.VideoCapture(os.path.join(dataset_path, folder, video_sequence))
            ret, image = video_cap.read()
            
            skip_counter = 0
            process = True # Videos are recorder at 120 Hz, so downsaple and read 1/4 frames

            while ret:
                if process:
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
                if skip_counter < 3:
                    process = False
                    skip_counter += 1
                
                elif skip_counter == 3:
                    process = True
                    skip_counter = 0
                
            skip = int(sequence_length - config_params['dataset_processor_params']['overlapping_frame_window']) # Set how many frames to skip when moving to next sequence

            # At the end of a single experiment, check all extracted sequence and decide if consistent or not
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
    dump_json(data, config_params['dataset_processor_params']['out_path'])       



def dump_json(data, out_path):

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


def load_config(file_path):
    with open(file_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as e:
            print(f"Error loading YAML file: {e}")
            return None


if __name__ == "__main__":
    main()