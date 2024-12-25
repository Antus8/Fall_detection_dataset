import cv2
import numpy as np
import mediapipe as mp

class LandmarkExtractor():
    def __init__(self):
        self.frame = None
        self.image_size = None

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=0.70)


    def draw_pose_landmarks(self, landmarks, image):
        self.mp_drawing.draw_landmarks(image, landmarks, self.mp_holistic.POSE_CONNECTIONS,
                                       self.mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2),
                                       self.mp_drawing.DrawingSpec(color=(180,100,100), thickness=2, circle_radius=2))

    def draw_hand_landmarks(self, landmarks, image):
        self.mp_drawing.draw_landmarks(image, landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                       self.mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2),
                                       self.mp_drawing.DrawingSpec(color=(180,100,100), thickness=2, circle_radius=2))
        
    def draw_selected_landmarks(self, landmarks, image):
        for key in landmarks.keys():
            cv2.circle(image, (landmarks[key][0], landmarks[key][1]), 4, (0, 0,255), thickness=-1)

        return image
        
    
    def get_body_landmarks(self, image):
        self.image_size = [image.shape[1], image.shape[0]]
        self.frame = image

        results = self.holistic.process(image)

        body_pose_landmarks = results.pose_landmarks
        right_hand_landmarks = results.right_hand_landmarks
        left_hand_landmarks = results.left_hand_landmarks

        normalized_landmarks = self.get_selected_body_landmarks(body_pose_landmarks)

        if normalized_landmarks is not None:
            right_wrist = self.get_wrist_landmarks(right_hand_landmarks)
            left_wrist = self.get_wrist_landmarks(left_hand_landmarks)

            normalized_landmarks['left_wrist'] = left_wrist
            normalized_landmarks['right_wrist'] = right_wrist
            
            img_landmarks = self.de_normalize_body_landmarks(normalized_landmarks)
            vd, hd = self.get_body_aspect_ratio(img_landmarks)

            if hd == 0:
                hd = 1
                print(f"Adjusting BAR {vd}/{hd}")                      # assume pixel-wise body width
            body_aspect_ratio = vd/hd       #TODO: to be normalized after dataset creation

            # image_out = image.copy()
            # image_out = self.draw_selected_landmarks(img_landmarks, image)
            # image_out = self.check_body_aspect_ratio(vd, hd, img_landmarks)
            # cv2.imshow('', image_out)
            # cv2.waitKey(20)

            return normalized_landmarks, body_aspect_ratio
        
        else:
            return None, None


            
    def check_body_landmarks(self, coords):
        for key in coords.keys():
            for landmark in coords[key]:
                if coords[key][0] > 1:
                    coords[key][0] = 0
                if coords[key][1] > 1:
                    coords[key][1] = 0


    def get_wrist_landmarks(self, hand_landmarks):
        if hand_landmarks is not None:
            for idx, landmark in enumerate(hand_landmarks.landmark):
                if idx == 0:
                    return[landmark.x, landmark.y]
                
        return [0, 0]


    def get_selected_body_landmarks(self, body_landmarks):
        coords = {'front_face':None, 'left_shoulder':None, 'right_shoulder':None, 'left_hip':None, 'right_hip':None, 'left_knee':None, 'right_knee':None, 'right_ankle':None, 'left_ankle':None}

        if body_landmarks is not None:
            for idx, landmark in enumerate(body_landmarks.landmark):
                if idx == 0:
                    coords['front_face'] = [landmark.x, landmark.y]
                elif idx == 11:
                    coords['left_shoulder'] = [landmark.x, landmark.y]
                elif idx == 12:
                    coords['right_shoulder'] = [landmark.x, landmark.y]
                elif idx == 23:
                    coords['left_hip'] = [landmark.x, landmark.y]
                elif idx == 24:
                    coords['right_hip'] = [landmark.x, landmark.y]
                elif idx == 25:
                    coords['left_knee'] = [landmark.x, landmark.y]
                elif idx == 26:
                    coords['right_knee'] = [landmark.x, landmark.y]
                elif idx == 28:
                    coords['right_ankle'] = [landmark.x, landmark.y]
                elif idx == 27:
                    coords['left_ankle'] = [landmark.x, landmark.y]

            return coords
        else: 
            return None
        
    
    def de_normalize_body_landmarks(self, coords):
        img_coords = {'front_face':None, 'left_shoulder':None, 'right_shoulder':None, 'left_hip':None, 'right_hip':None, 'left_knee':None, 'right_knee':None, 'right_ankle':None, 'left_ankle':None}

        for key in coords.keys():
            img_coords[key] =  [int(coords[key][0]*self.image_size[0]), int(coords[key][1]*self.image_size[1])]

        return img_coords
    

    def get_body_aspect_ratio(self, normalized_landmarks):
        '''
        Here I compute vertical max distance between shoulders and knees, as well as their horizontal distance
        If the ratio vertical/horizontal is > 1 I am in a STANDING position, otherwise I am prone
        '''
        vertical_distance = max(abs(normalized_landmarks['left_shoulder'][1] - normalized_landmarks['right_knee'][1]), abs(normalized_landmarks['right_shoulder'][1] - normalized_landmarks['left_knee'][1]))
        
        horizontal_distance = max(abs(normalized_landmarks['left_shoulder'][0] - normalized_landmarks['right_knee'][0]), abs(normalized_landmarks['right_shoulder'][0] - normalized_landmarks['left_knee'][0]))
        

        return vertical_distance, horizontal_distance
    

    def check_body_aspect_ratio(self, vd, hd, img_landmarks):
        image_out = self.frame.copy()
        center_shoulders = [int((img_landmarks['left_shoulder'][0] + img_landmarks['right_shoulder'][0]) //2), int(img_landmarks['left_shoulder'][1])]
        end_shoulders = [int((img_landmarks['left_shoulder'][0] + img_landmarks['right_shoulder'][0]) //2), int(img_landmarks['left_shoulder'][1])+vd]
        end_body = [int(img_landmarks['right_shoulder'][0] + hd),  int((img_landmarks['right_shoulder'][1] + img_landmarks['right_knee'][1]) //2)]
        center_body = [int(img_landmarks['right_shoulder'][0]), int((img_landmarks['right_shoulder'][1] + img_landmarks['right_knee'][1]) //2)]

        print(center_shoulders)
        print(center_body)
        cv2.line(image_out,  center_shoulders, end_shoulders, (0,0,255), 3)
        cv2.line(image_out, center_body, end_body, (0,0,255), 3)

        return image_out
