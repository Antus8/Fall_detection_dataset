import cv2
import json
import random
import numpy as np
import mediapipe as mp
from landmark_extractor import LandmarkExtractor



class DatasetCreator:
    def __init__(self):
        self.camera_channel = cv2.VideoCapture(0)
        self.landmark_extractor = LandmarkExtractor()
        self.landmarks_keys = ['front_face','left_wrist','right_wrist','left_shoulder','right_shoulder','left_hip','right_hip','left_knee','right_knee','right_ankle','left_ankle']

        self.fps = self.camera_channel.get(cv2.CAP_PROP_FPS)
        self.coverage = 3 # set sequence length in seconds
        self.sequence_length = int(self.fps * self.coverage)

        self.queue = []
        self.labels = []

        self.frames_counter = 0 # Set video length to be packed in the dataset
        self.out_path = "~/Fall_detection_dataset/Dataset Tools/out/custom_dataset.json"
        self.images_queue = []



    def start(self):
        while(self.camera_channel.isOpened()):

            _, frame = self.camera_channel.read()

            self.image_size = [frame.shape[1], frame.shape[0]]

            body_landmarks, body_ar = self.landmark_extractor.get_body_landmarks(frame)

            # self.display_landmarks(body_landmarks, frame)
            if body_landmarks:
                if self.check_landmarks(body_landmarks):
                    feature_vector = self.vectorize_landmarks(body_landmarks, body_ar)
                    self.queue.append(feature_vector)
                    self.labels.append(random.randint(0,1))
                    self.images_queue.append(frame)

                    self.frames_counter += 1
                    print(f"Processed frame {self.frames_counter}/100....")
                    if self.frames_counter == 1000:
                        break

        self.dump_dataset()
        # self.check()



    def check(self):
        with open(self.out_path, 'r') as json_file:
            data = json.load(json_file)

        i = 0
        sample = data[0]
        for feature_vector in sample:
            out_img = self.images_queue[i].copy()
            i = i+1

            feature_vector = feature_vector[:22] # ignore body aspect ratio
            for c in range(0, len(feature_vector), 2):
                feature_vector[c] = int(feature_vector[c]*self.image_size[0])
                feature_vector[c+1] = int(feature_vector[c+1]*self.image_size[1])

            
                cv2.circle(out_img, (feature_vector[c], feature_vector[c+1]), 4, (0, 0,255), thickness=-1)

            cv2.imshow('', out_img)
            cv2.waitKey(0)
            
                
                
    def dump_dataset(self):
        '''
        Export to json file
        PyTorch DataLoader will load data as an array where each element is a sample related to a frame and process it frame by frame
        Each sample has 23 features (11 x,y body landmarks + body aspect ratio) and a label
        Shift window by a factor of 15 frames
        '''
        data = []        

        # Write data rows
        for i in range(0, len(self.queue) - self.sequence_length + 1, int(self.fps/2)):
            print(f"writing strting from i {i}")
            # Flatten the sequence of feature vectors into a single row
            row = []
            for j in range(self.sequence_length):
                row.append(self.queue[i + j])
            
            print(f"Writing from {i} to {i+j}")
            
            if self.labels[i+j] is not None:
                row.append(self.labels[i+j])
            
            data.append(row)

        with open(self.out_path, 'w') as json_file:
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

            
    def vectorize_landmarks(self, landmarks, body_ar):
        input_vector = []
        for key in self.landmarks_keys:
            input_vector.append(float(landmarks[key][0]))
            input_vector.append(float(landmarks[key][1]))

        input_vector.append(body_ar)
        return input_vector



    def display_landmarks(self, body_landmarks, image):
        normal_bl = self.landmark_extractor.de_normalize_body_landmarks(body_landmarks)
        out_img = self.landmark_extractor.draw_selected_landmarks(normal_bl, image)
        cv2.imshow('', out_img)
        cv2.waitKey(20) == 27
            


    def check_landmarks(self, body_landmarks):
        for key in body_landmarks:
            if(body_landmarks[key][0]<=0 or body_landmarks[key][0]>1 or body_landmarks[key][1]<=0 or body_landmarks[key][1]>1):
                return False
        return True



if __name__ == "__main__":
    dataset_creator = DatasetCreator()
    dataset_creator.start()