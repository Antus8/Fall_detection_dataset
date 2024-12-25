import os
import json
import random

def main():
    input_directory = "~/Fall_detection_dataset/Dataset Tools/Data_Files"
    out_path = "~/Fall_detection_dataset/Dataset Tools/out/final_merged_dataset.json"
    global_dataset = []
    tot_samples = 0

    for file in os.listdir(input_directory):
        file_path = os.path.join(input_directory, file)

        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        tot_samples = tot_samples + len(data)
        previous_bar = None

        for sample in data:
            for frame_data in sample[:-1]:
                # Adjusting Body aspect ratio 

                if frame_data[22] is None:
                    frame_data[22] = previous_bar
                elif frame_data[22] > 40:
                    frame_data[22] = frame_data[22]/10 # suppose a 10 px body width 
                
                previous_bar = frame_data[22]

                # if (frame_data[22] < 0) or  (frame_data[22] > 40) or (frame_data[22] is None):
                #     print("Error")
        
            global_dataset.append(sample)
    
    if len(global_dataset) == tot_samples:
        random.shuffle(global_dataset) # perform shuffling
        random.shuffle(global_dataset)
        
        # Check dimensions after shuffling
        print(len(global_dataset))
        print(len(global_dataset[0]))
        print(len(global_dataset[0][0]))

        dump_merged_dataset(global_dataset, out_path)
        print("Ok...")


def dump_merged_dataset(data, out_path):
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



if __name__ == "__main__":
    main()
