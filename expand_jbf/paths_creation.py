import os

"""
if the code to create the path files in expand_radar.py is not working, run this
"""

current_dir = os.path.dirname(os.path.abspath(__file__))

# full dataset output in thesis_code folder
nuscenes_data_dir = r'/media/bourcha/F6C0FAB4C0FA79E7/bourcha/Nuscenes-dataset-full'
base_output_dir = os.path.join(nuscenes_data_dir, 'data_split')

if not os.path.exists("paths"):
    os.makedirs("paths")

''' Functions to create path txt files'''
def list_files_in_directory(directory):
    files_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            files_list.append(os.path.join(root, file))
    return files_list

def create_file_paths(base_output_dir, main_folder, parent_dir):
    main_dir = os.path.join(base_output_dir, main_folder)
    paths_dir = os.path.join(parent_dir, 'paths')

    for sub_dir in os.listdir(main_dir):
        sub_dir_path = os.path.join(main_dir, sub_dir)
        if os.path.isdir(sub_dir_path):
            output_file = os.path.join(paths_dir, f'nuscenes_{main_folder}_{sub_dir}.txt')
            with open(output_file, 'w') as f:
                files = list_files_in_directory(sub_dir_path)
                for file_path in files:
                    f.write(f"{file_path}\n")
        print(output_file)

create_file_paths(base_output_dir, 'train', current_dir)
print("Done with training paths")
create_file_paths(base_output_dir, 'validation', current_dir)
print("Done with validation paths")

print()