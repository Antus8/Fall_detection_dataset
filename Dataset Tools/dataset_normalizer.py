import json
import numpy as np




class DatasetNormalizer:
    def __init__(self):
        self.in_path = "~/Fall_detection_dataset/Dataset Tools/out/final_merged_dataset.json"
        self.out_path = "~/Fall_detection_dataset/Dataset Tools/out/final_normalized_dataset.json"

    def normalize(self):
        with open(self.in_path, 'r') as json_file:
            data = json.load(json_file)

        min_bar, max_bar = self.extract_min_max(data)
        print(min_bar)
        print(max_bar)
        
        for row in data:
            for vector in row[:-1]:  # Again, exclude the label if present
                value = vector[22]
                normalized_value = (value - min_bar) / (max_bar - min_bar)
                vector[22] = normalized_value

        self.dump_dataset(data)

    

    def check_normalized_dataset(self):
        with open(self.out_path, 'r') as json_file:
            data = json.load(json_file)

        print(len(data))
        print(len(data[0]))
        print(len(data[0][0]))

        for row in data:
            for vector in row[:-1]:
                for el in vector:
                    if (el < 0) or (el > 1):
                        print(el)
                        return False

        return True




    def dump_dataset(self, data):
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
        

        
    def extract_min_max(self, data):
        min_value = float('inf')
        max_value = float('-inf')

        for row in data:
            for vector in row[:-1]:  # Exclude the last element if it's the label
                bar = vector[22]  # 23rd element (0-indexed, so it's index 22)
                if bar < min_value:
                    min_value = bar
                if bar > max_value:
                    max_value = bar
        
        return min_value, max_value



if __name__ == "__main__":
    dataset_normalizer = DatasetNormalizer()
    # dataset_normalizer.normalize()
    print(dataset_normalizer.check_normalized_dataset())
        