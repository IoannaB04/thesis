import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from matplotlib.colors import Normalize
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.splits import create_splits_scenes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from nuscenes.utils.data_classes import RadarPointCloud
import time


def plot_array(array, image):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis('off')
    ax.imshow(image)

    non_zero_indices = np.nonzero(array)
    non_zero_depths = array[non_zero_indices]

    scatter = ax.scatter(non_zero_indices[1], non_zero_indices[0], c=non_zero_depths, cmap='jet', s=5)
    vmin, vmax = np.min(non_zero_depths), np.max(non_zero_depths)
    scatter.set_norm(Normalize(vmin=vmin, vmax=vmax))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(scatter, cax=cax)
    cbar.set_label('Depth (meters)')

    # cbar.ax.yaxis.set_ticks_position('left')
    cbar.set_ticks(np.linspace(vmin, vmax, num=5))

    ax.set_title('Filtered radar points', fontsize=16)

    plt.show()
    plt.close()

number_of_total_sweeps = 5
sigma_s = 25
sigma_r = 5

# nuscenes_data_dir = r'/media/bourcha/F6C0FAB4C0FA79E7/bourcha/Nuscenes-dataset-full'
# nusc = NuScenes(version='v1.0-trainval', dataroot=nuscenes_data_dir, verbose=True)

nuscenes_data_dir = r'C:\Users\Black Pearl\Desktop\thesis_code\downloaded_dataset'
nusc = NuScenes(version='v1.0-mini', dataroot=nuscenes_data_dir, verbose=True)


# Define output directories
current_dir = os.path.dirname(os.path.abspath(__file__))
base_output_dir = os.path.join(current_dir, 'jbf_r4dyn') # where the jbf output is saved

train_dir = os.path.join(base_output_dir, 'train')
validation_dir = os.path.join(base_output_dir, 'validation')

# Splitting the scenes to train and validation with the predefined splits of nuScenes
splits = create_splits_scenes()
train_scenes = splits['train']
val_scenes = splits['val']
print(f'Train samples: {len(train_scenes)}')
print(f'Validation samples: {len(val_scenes)}')

# if not os.path.exists("paths"):
#     os.makedirs("paths")
#
# # paths for rgb images in nuscenes, need them for fusionnet
# train_filename = os.path.join("paths", "nuscenes_train_image.txt")
# val_filename = os.path.join("paths", "nuscenes_validation_image.txt")

def create_output_directories(base_dir, scene_name):
    sub_dirs = [
        'depth_map',
        'confidence_map',
        'visualization',
        'confidence_map_heatmap',
        'radar_on_image',
        'radar_on_image_filtered'
    ]

    directories = {}

    for sub_dir in sub_dirs:
        dir_path = os.path.join(base_dir, sub_dir, scene_name)
        os.makedirs(dir_path, exist_ok=True)
        directories[sub_dir] = dir_path

    return directories

def plot_image(type_file, data, name, scene_dir, image=None, colorbar=False):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis('off')

    if type_file == 'radar_on_image':
        ax.imshow(image)
        scatter = ax.scatter(data[:, 0], data[:, 1], c=data[:, 2], cmap=cm.jet, s=5)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(scatter, cax=cax)
        cbar_label = 'Depth (meters)'
        cbar.set_label(cbar_label)

        file_suffix = '_radar_on_image.png'

    elif type_file == 'expanded_depth':
        plt.close()
        img = Image.fromarray(data.astype(np.uint8), mode='L')
        output_path = os.path.join(scene_dir, name + '_depth_map.png')
        img.save(output_path)
        return

    elif type_file == 'confidence_map':
        ax.imshow(data, cmap='hot')
        if colorbar:
            scatter = cm.ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap='hot')

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(scatter, cax=cax)
            cbar_label = 'Confidence'
            cbar.set_label(cbar_label)

            file_suffix = '_confidence_map_colorbar.png'
        else:
            file_suffix = '_confidence_map.png'

    elif type_file == 'visualization' or 'radar_on_image_filtered':
        point_size_mapping = {
            'visualization': 0.01,
            'radar_on_image_filtered': 5
        }

        ax.imshow(image)
        non_zero_indices = np.nonzero(data)
        non_zero_depths = data[non_zero_indices]
        scatter = ax.scatter(non_zero_indices[1], non_zero_indices[0], c=non_zero_depths, cmap='jet', s=point_size_mapping.get(type_file))
        vmin, vmax = np.min(non_zero_depths), np.max(non_zero_depths)
        scatter.set_norm(Normalize(vmin=vmin, vmax=vmax))

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(scatter, cax=cax)
        cbar.set_label('Depth (meters)')
        cbar.set_ticks(np.linspace(vmin, vmax, num=5))

        file_suffix = '.png'

    else:
        raise ValueError(f"Unknown plot type: {type_file}")

    output_path = os.path.join(scene_dir, name + file_suffix)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()



LUT_s = np.exp(-0.5 * (np.arange(500) ** 2) / sigma_s ** 2)
LUT_r = np.exp(-0.5 * (np.arange(256) ** 2) / sigma_r ** 2)
MAX_SHIFT = next((i for i, j in enumerate(LUT_s < 0.1) if j), None)

def jbf_method(radar_depth, image, f_u, f_v, LUT_s=LUT_s, LUT_r=LUT_r, MAX_SHIFT=MAX_SHIFT, w=1, h=2):
    init_depth = radar_depth
    ex_depth = radar_depth
    ex_conf_score = np.zeros(radar_depth.shape)
    height, width, _ = image.shape

    Y, X = np.where(init_depth > 0)
    D = init_depth[Y, X]
    sort_index = D.argsort()

    for _index in sort_index:
        p_x = X[_index]
        p_y = Y[_index]
        p_d = D[_index]
        p_i = image[p_y, p_x]

        ex_conf_score[p_y, p_x] = 1

        # Get relative window for each radar point
        v = int((h * f_v) / p_d)
        u = int((w * f_u) / p_d)
        dv = int(np.min([MAX_SHIFT, int(v / 2)]))
        du = int(np.min([MAX_SHIFT, int(u / 2)]))

        for i in range(-du, du):
            for j in range(-dv, dv):
                q_y = np.max([0, np.min([height - 1, p_y + j])])
                q_x = np.max([0, np.min([width - 1, p_x + i])])
                q_i = image[q_y, q_x]

                if init_depth[q_y, q_x]: continue

                d_x = np.abs(q_x - p_x)
                d_y = np.abs(q_y - p_y)
                d_r = np.abs(int(p_i[0]) - int(q_i[0]))
                d_g = np.abs(int(p_i[1]) - int(q_i[1]))
                d_b = np.abs(int(p_i[2]) - int(q_i[2]))

                G_s = LUT_s[d_x] * LUT_s[d_y]
                G_i = LUT_r[d_r] * LUT_r[d_g] * LUT_r[d_b]
                G_jbf = G_s * G_i

                if G_jbf > 0.05:
                    ex_conf_score[q_y, q_x] = G_jbf
                    ex_depth[q_y, q_x] = p_d

    return ex_depth, ex_conf_score


def bounding_boxes_binary_masks_bottom_half(image_shape, boxes, camera_intrinsic):
    img_height, img_width = image_shape[:2]
    bb_array = np.zeros((img_height, img_width, len(boxes)), dtype=np.uint8)

    for i, box in enumerate(boxes):
        corners_3d = box.corners()
        corners_2d = view_points(corners_3d, camera_intrinsic, normalize=True)[:2, :].T
        corners_2d = np.int32(corners_2d)

        y_up   = np.min(corners_2d[7,1])
        y_down = np.max(corners_2d[4,1])
        height = y_up - y_down

        rows_to_divide = [0, 1, 5, 4] # Top face
        corners_2d[rows_to_divide, 1] = corners_2d[rows_to_divide, 1] + height/2

        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        faces = [
            corners_2d[:4],  # Front face
            corners_2d[4:],  # Back face
            corners_2d[[0, 1, 5, 4]],  # Top face
            corners_2d[[2, 3, 7, 6]],  # Down face
            corners_2d[[0, 3, 7, 4]],  # Left face
            corners_2d[[1, 2, 6, 5]]   # Right face
        ]

        # Fill the faces
        for face in faces:
            cv2.fillConvexPoly(mask, face, True)

        # Fill the interior of the bounding box
        bb_array[:, :, i] = mask

        # Store the bb_array in the corresponding bounding box
        box.bb_array_mask = bb_array[:, :, i]

    final_mask = np.sum(bb_array, axis=2)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(final_mask, cmap='gray')
    # plt.title('Bottom half of bounding boxes')
    # plt.axis('off')
    # plt.show()



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

    # scene_number = int(scene_name.split('-')[1])  # scene names are in the format "scene-XXXX" for nuscene dataset
    # if scene_number < 1109 or scene_name in val_scenes:
    #     continue  # Skip scenes before 1080

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
            # with open(train_filename, "a") as train_file:
            #     train_file.write(camera_filepath + "\n")
        elif scene_name in val_scenes:
            base_dir = validation_dir
            # with open(val_filename, "a") as val_file:
            #     val_file.write(camera_filepath + "\n")
        else:
            print(f"\nScene {scene_name} not found in train or validation lists.")
            continue

        print(f"\rScene {scene_name} : {current_sample_counter}/{total_samples} \t{base_dir}", end='', flush=True)

        directories = create_output_directories(base_dir, scene_name)

        radar_token = my_sample['data']['RADAR_FRONT']
        sweeps = []

        while radar_token and len(sweeps) < number_of_total_sweeps:
            radar_data = nusc.get('sample_data', radar_token)
            sweeps.append(radar_data)
            radar_token = radar_data['prev']

        all_radar_points = []
        for radar_data in reversed(sweeps):
            radar_points = RadarPointCloud.from_file(os.path.join(nuscenes_data_dir, radar_data['filename']))
            radar_pose_rec = nusc.get('ego_pose', radar_data['ego_pose_token'])
            radar_pose = Quaternion(radar_pose_rec['rotation']).rotation_matrix
            radar_points.rotate(radar_pose)
            radar_points.translate(np.array(radar_pose_rec['translation']))

            cam_pose_rec = nusc.get('ego_pose', camera_data['ego_pose_token'])
            cam_pose = Quaternion(cam_pose_rec['rotation']).rotation_matrix
            cam_translation = np.array(cam_pose_rec['translation'])
            inv_cam_pose = np.linalg.inv(cam_pose)
            radar_points.translate(-cam_translation)
            radar_points.rotate(inv_cam_pose)

            ego_pose_at_cam_time = nusc.get('ego_pose', camera_data['ego_pose_token'])
            ego_pose_at_radar_time = nusc.get('ego_pose', radar_data['ego_pose_token'])
            radar_to_cam_translation = np.array(ego_pose_at_cam_time['translation']) - np.array(
                ego_pose_at_radar_time['translation'])
            rotation_diff = Quaternion(ego_pose_at_cam_time['rotation']) * Quaternion(
                ego_pose_at_radar_time['rotation']).inverse
            radar_to_cam_rotation = rotation_diff.rotation_matrix

            radar_points.rotate(radar_to_cam_rotation.T)
            radar_points.translate(-radar_to_cam_translation)

            points, coloring, _ = nusc.explorer.map_pointcloud_to_image(radar_data['token'], camera_data['token'])
            valid_indices = (points[0, :] >= 0) & (points[0, :] < camera_image.width) & (points[1, :] >= 0) & (
                        points[1, :] < camera_image.height)
            valid_points = points[:, valid_indices]
            valid_distances = coloring[valid_indices]

            all_radar_points.extend(zip(valid_points[0], valid_points[1], valid_distances))

        all_radar_points = np.array(all_radar_points)

        ''' Filtering the radar points using the 2D bounding boxes and tolerance d < dm + β '''
        # Get the boxes with filtered annotations
        _, boxes, _ = nusc.get_sample_data(my_sample['data']['CAM_FRONT'])
        anns = [nusc.get('sample_annotation', ann_token) for ann_token in my_sample['anns']]
        filtered_anns = [ann for ann in anns if ann['visibility_token'] >= '4']
        ''' 1: 0-40%   (invisible)
            2: 40-60%  (partially visible)
            3: 60-80%  (mostly visible)
            4: 80-100% (fully visible)'''
        filtered_boxes = [box for box in boxes if box.token in [ann['token'] for ann in filtered_anns]]
        del boxes, filtered_anns

        print()

        cam_sensor = nusc.get('sample_data', my_sample['data']['CAM_FRONT'])
        calibrated_sensor = nusc.get('calibrated_sensor', cam_sensor['calibrated_sensor_token'])
        camera_intrinsic = np.array(calibrated_sensor['camera_intrinsic'])

        bounding_boxes_binary_masks_bottom_half(camera_image_np.shape, filtered_boxes, camera_intrinsic)

        # all_radar_points to image plane in numpy format
        radar_depth = np.zeros(camera_image_np.shape[:2], dtype=np.float32)
        for idx, (x, y) in enumerate(zip(all_radar_points[:, 0], all_radar_points[:, 1])):
            # if 0 <= x < radar_depth.shape[1] and 0 <= y < radar_depth.shape[0]:
                radar_depth[int(y), int(x)] = max(radar_depth[int(y), int(x)], all_radar_points[idx, 2])

        filtered_radar_points = np.zeros(camera_image_np.shape[:2], dtype=np.float32)

        epsilon = 1e-6  # to avoid division by zero
        for box in filtered_boxes:
            # keeping radar points at the bottom half of the bounding box
            radar_in_box = radar_depth * box.bb_array_mask

            # calculating the weighted harmonic mean
            non_zero_elements = radar_in_box[radar_in_box != 0]
            if len(non_zero_elements) > 0:
                min_value = non_zero_elements.min()

                # calculation of weighting factors
                weights = 1 / (epsilon + np.abs(non_zero_elements - min_value))

                # weighted harmonic mean
                weighted_harmonic_mean = np.sum(weights) / np.sum(weights / non_zero_elements)

                # applying depth threshold to address the noise
                depth_threshold = min_value + weighted_harmonic_mean
                mask_above_threshold = (radar_in_box > 0) & (radar_in_box <= depth_threshold)

                filtered_radar_points[mask_above_threshold] = radar_in_box[mask_above_threshold]

                # print(non_zero_elements.min(), '\t', non_zero_elements.max(), '\t', harmonic_mean)
        # plot_array(array=filtered_radar_points, image=camera_image_np)

        name = os.path.basename(camera_filepath)[:-4]

        ''' PLOT RADAR POINTS ON TOP OF RGB IMAGE '''
        plot_image('radar_on_image', all_radar_points, name, directories['radar_on_image'], image=camera_image_np)

        ''' PLOT FILTERED RADAR POINTS ON TOP OF RGB IMAGE '''
        plot_image('radar_on_image_filtered', filtered_radar_points, name, directories['radar_on_image_filtered'], image=camera_image_np)


        ''' APPLY JOINT BILATERAL FILTERS'''
        # Get camera intrinsic values
        f_u = camera_intrinsic[0, 0]
        f_v = camera_intrinsic[1, 1]

        # Apply the JBF method
        expanded_depth_map, confidence_map = jbf_method(filtered_radar_points, camera_image_np, f_u, f_v)

        # Save confidence map as numpy array for FusionNET
        confidence_map_path = os.path.join(directories['confidence_map'], name + "_confidence.npy")
        np.save(confidence_map_path, confidence_map)

        # Save confidence map as heatmap
        plot_image('confidence_map', confidence_map, name, directories['confidence_map_heatmap'])
        # plot_image('confidence_map', confidence_map, name, directories['confidence_map_heatmap_colorbar'], colorbar=True)

        # Save expanded depth map
        plot_image('expanded_depth', expanded_depth_map, name, directories['depth_map'])

        # Save image with expanded depth map on top
        plot_image('visualization', expanded_depth_map, name, directories['visualization'], image=camera_image_np)

    end_time = time.time()  # End time for the scene processing
    elapsed_time = end_time - start_time
    print(f"\nProcessing time for scene {scene_name}: {elapsed_time:.2f} seconds\n")
