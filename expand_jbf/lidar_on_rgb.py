import os
import numpy as np
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud
import time

number_of_total_sweeps = 1

# Update the path to the nuscenes dataset directory
nuscenes_data_dir = r'/media/bourcha/F6C0FAB4C0FA79E7/bourcha/Nuscenes-dataset-full'
nusc = NuScenes(version='v1.0-trainval', dataroot=nuscenes_data_dir, verbose=True)


# Define output directories
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Full dataset output in thesis_code folder
base_output_dir = os.path.join(nuscenes_data_dir, 'data_split')
train_dir = os.path.join(base_output_dir, 'train')
validation_dir = os.path.join(base_output_dir, 'validation')

# Splitting the scenes to train and validation with the predefined splits of nuScenes
splits = create_splits_scenes()
train_scenes = splits['train']
val_scenes = splits['val']
print(f'Train samples: {len(train_scenes)}')
print(f'Validation samples: {len(val_scenes)}')

if not os.path.exists("paths"):
    os.makedirs("paths")

def create_output_directories(base_dir, scene_name):
    sub_dirs = ['lidar']

    directories = {}

    for sub_dir in sub_dirs:
        dir_path = os.path.join(base_dir, sub_dir, scene_name)
        os.makedirs(dir_path, exist_ok=True)
        directories[sub_dir] = dir_path

    return directories

def plot_image(data, name, scene_dir, image):
    height, width = image.shape[:2]
    data_projection = np.zeros((height, width))

    for point in data:
        x,y,z = int(point[0]), int(point[1]), point[2]
        if 0 <= x <= width and 0 <= y <=height:
            data_projection[y,x] = z

    img = Image.fromarray(data_projection.astype(np.uint8), mode='L')
    output_path = os.path.join(scene_dir, name + '_lidar_depth_map.png')
    img.save(output_path)


# Grouping of samples by scene
samples_by_scene = {}
for my_sample in nusc.sample:
    scene_token = my_sample['scene_token']
    if scene_token not in samples_by_scene:
        samples_by_scene[scene_token] = []
    samples_by_scene[scene_token].append(my_sample)
print()

# Edit samples per scene
for scene_token, scene_samples in samples_by_scene.items():
    scene = nusc.get('scene', scene_token)
    scene_name = scene['name']

    total_samples = len(scene_samples)
    current_sample_counter = 0

    start_time = time.time()  # Start time for the scene processing

    for my_sample in scene_samples:
        camera_data = nusc.get('sample_data', my_sample['data']['CAM_FRONT'])
        camera_filepath = os.path.join(nuscenes_data_dir, camera_data['filename'])
        camera_image = Image.open(camera_filepath)
        camera_image_np = np.array(camera_image)

        current_sample_counter += 1

        if scene_name in train_scenes:
            base_dir = train_dir
        elif scene_name in val_scenes:
            base_dir = validation_dir
        else:
            print(f"\nScene {scene_name} not found in train or validation lists.")
            continue

        print(f"\rScene {scene_name} : {current_sample_counter}/{total_samples} \t{base_dir}", end='', flush=True)

        directories = create_output_directories(base_dir, scene_name)

        lidar_token = my_sample['data']['LIDAR_TOP']
        sweeps = []

        while lidar_token and len(sweeps) < number_of_total_sweeps:
            lidar_data = nusc.get('sample_data', lidar_token)
            sweeps.append(lidar_data)
            lidar_token = lidar_data['prev']

        all_lidar_points = []
        for lidar_data in reversed(sweeps):
            lidar_points = LidarPointCloud.from_file(os.path.join(nuscenes_data_dir, lidar_data['filename']))
            lidar_pose_rec = nusc.get('ego_pose', lidar_data['ego_pose_token'])
            lidar_pose = Quaternion(lidar_pose_rec['rotation']).rotation_matrix
            lidar_points.rotate(lidar_pose)
            lidar_points.translate(np.array(lidar_pose_rec['translation']))

            cam_pose_rec = nusc.get('ego_pose', camera_data['ego_pose_token'])
            cam_pose = Quaternion(cam_pose_rec['rotation']).rotation_matrix
            cam_translation = np.array(cam_pose_rec['translation'])
            inv_cam_pose = np.linalg.inv(cam_pose)
            lidar_points.translate(-cam_translation)
            lidar_points.rotate(inv_cam_pose)

            ego_pose_at_cam_time = nusc.get('ego_pose', camera_data['ego_pose_token'])
            ego_pose_at_lidar_time = nusc.get('ego_pose', lidar_data['ego_pose_token'])
            lidar_to_cam_translation = np.array(ego_pose_at_cam_time['translation']) - np.array(
                ego_pose_at_lidar_time['translation'])
            rotation_diff = Quaternion(ego_pose_at_cam_time['rotation']) * Quaternion(
                ego_pose_at_lidar_time['rotation']).inverse
            lidar_to_cam_rotation = rotation_diff.rotation_matrix

            lidar_points.rotate(lidar_to_cam_rotation.T)
            lidar_points.translate(-lidar_to_cam_translation)

            points, coloring, _ = nusc.explorer.map_pointcloud_to_image(lidar_data['token'], camera_data['token'])
            valid_indices = (points[0, :] >= 0) & (points[0, :] < camera_image.width) & (points[1, :] >= 0) & (points[1, :] < camera_image.height)
            valid_points = points[:, valid_indices]
            valid_distances = coloring[valid_indices]

            all_lidar_points.extend(zip(valid_points[0], valid_points[1], valid_distances))
            valid_indices = (points[0, :] >= 0) & (points[0, :] < camera_image.width) & (points[1, :] >= 0) & (points[1, :] < camera_image.height)
            valid_points = points[:, valid_indices]
            valid_distances = coloring[valid_indices]

            all_lidar_points.extend(zip(valid_points[0], valid_points[1], valid_distances))

        all_lidar_points = np.array(all_lidar_points)

        name = os.path.basename(camera_filepath)[:-4]
        ''' PLOT LIDAR POINTS ON TOP OF RGB IMAGE '''
        plot_image(all_lidar_points, name, directories['lidar'], image=camera_image_np)


    end_time = time.time()  # End time for the scene processing
    elapsed_time = end_time - start_time
    print(f"\nProcessing time for scene {scene_name}: {elapsed_time:.2f} seconds\n")
