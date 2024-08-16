import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from matplotlib.colors import Normalize
from nuscenes.utils.geometry_utils import view_points
from mpl_toolkits.axes_grid1 import make_axes_locatable


nuscenes_data_dir = r'/imec/users/bourch01/dl4ms/v1.0-mini/'
nusc = NuScenes(version='v1.0-mini', dataroot=nuscenes_data_dir, verbose=True)

# base_output_dir = os.path.join('labels')
# if not os.path.exists(base_output_dir):
#     os.makedirs(base_output_dir)

# # Splitting the scenes to train and validation with the predefined splits of nuScenes
# splits = create_splits_scenes()
# train_scenes = splits['train']
# val_scenes = splits['val']
# # print(f'Train samples: {len(train_scenes)}')
# # print(f'Validation samples: {len(val_scenes)}')

# # Define output directories
# train_dir = os.path.join(base_output_dir, 'train')
# validation_dir = os.path.join(base_output_dir, 'validation')

base_output_dir = os.path.join('labels', 'dokimi')
if not os.path.exists(base_output_dir):
    os.makedirs(base_output_dir)



scene = nusc.scene[0]
first_sample_token = scene['first_sample_token']

first_sample = nusc.get('sample', first_sample_token)
second_sample_token = first_sample['next']

sample = nusc.get('sample', second_sample_token)

# Select the front camera
cam_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
img = plt.imread(f"{nuscenes_data_dir}/{cam_data['filename']}")

# Load camera sensor and related data
cam_sensor = nusc.get('sample_data', sample['data']['CAM_FRONT'])
calibrated_sensor = nusc.get('calibrated_sensor', cam_sensor['calibrated_sensor_token'])
cam_intrinsic = np.array(calibrated_sensor['camera_intrinsic'])
ego_pose = nusc.get('ego_pose', cam_sensor['ego_pose_token'])

# Get the boxes with filtered annotations
_, boxes, _ = nusc.get_sample_data(sample['data']['CAM_FRONT'])
anns = [nusc.get('sample_annotation', ann_token) for ann_token in sample['anns']]
filtered_anns = [ann for ann in anns if ann['visibility_token'] >= '4']
''' 1: 0-40%   (invisible)
    2: 40-60%  (partially visible)
    3: 60-80%  (mostly visible)
    4: 80-100% (fully visible)'''

filtered_boxes = [box for box in boxes if box.token in [ann['token'] for ann in filtered_anns]]
del boxes


# def filter_boxes_depth(boxes, max_distance=90):
#     for i, box in enumerate(boxes):
#         corners_3d = box.corners()
#         ''' x, y, z
#               3-------2
#              /|      /|
#             7-------6 |
#             | |     | |
#             | 0-----|-1
#             |/      |/
#             4-------5
#         '''

#         # Calculate depth distances for the front and back plane
#         front_depth = int(np.max(corners_3d[2:, [4, 5, 6, 7]]))
#         back_depth  = int(np.min(corners_3d[2:, [0, 1, 3, 2]])) + front_depth

#         if back_depth > max_distance:
#             boxes.remove(box)

# filter_boxes_depth(filtered_boxes)


def bounding_boxes_binary_masks(image_shape, boxes, camera_intrinsic):
    img_height, img_width = image_shape[:2]
    bb_array = np.zeros((img_height, img_width, len(boxes)), dtype=np.uint8)

    for i, box in enumerate(boxes):
        corners_3d = box.corners()
        corners_2d = view_points(corners_3d, camera_intrinsic, normalize=True)[:2, :].T

        corners_2d = np.int32(corners_2d)

        mask = np.zeros((img_height, img_width), dtype=np.uint8)

        # Define faces
        faces = [
            corners_2d[:4],            # Front face
            corners_2d[4:],            # Back face
            corners_2d[[0, 1, 5, 4]],  # Left face
            corners_2d[[2, 3, 7, 6]],  # Right face
            corners_2d[[0, 3, 7, 4]],  # Top face
            corners_2d[[1, 2, 6, 5]]   # Bottom face
        ]

        # Fill the faces
        for face in faces:
            cv2.fillConvexPoly(mask, face, True)

        # Fill the interior of the bounding box
        bb_array[:, :, i] = mask

        # Store the bb_array in the corresponding bounding box
        box.bb_array_mask = bb_array[:, :, i]

def combine_masks(image_shape, filtered_boxes):
    # Initialize a blank mask with the same dimensions as the image
    combined_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    
    # Iterate over each bounding box and add the mask to the combined mask
    for box in filtered_boxes:
        combined_mask = np.maximum(combined_mask, box.bb_array_mask)
    
    return combined_mask


def z_buffer_code(image, depth_map_, boxes, camera_intrinsic, plot = False, scene_dir=None, name=None):
  
    # Create a binary mask for each boundng box
    bounding_boxes_binary_masks(image.shape, boxes, camera_intrinsic)
    binary_mask = combine_masks(image.shape, boxes)

    depth_map = binary_mask*depth_map_

    depth_map = depth_map.astype(float)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis('off')
    ax.imshow(image)
    non_zero_indices = np.nonzero(depth_map)
    non_zero_depths = depth_map[non_zero_indices]
    scatter = ax.scatter(non_zero_indices[1], non_zero_indices[0], c=non_zero_depths, cmap='jet', s=0.01)
    vmin, vmax = 0, np.max(non_zero_depths)
    scatter.set_norm(Normalize(vmin=vmin, vmax=vmax))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(scatter, cax=cax)
    cbar.set_label('Depth (meters)')
    output_path = os.path.join(scene_dir, "dok_0_bbbb.png")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    # plt.imsave(os.path.join(scene_dir, f"{name}.png"), depth_map)
    plt.close()

    print("Begin..")

    depth_map_final = np.zeros_like(depth_map, dtype=float)

    for i, box in enumerate(boxes):

        depth_map_box = depth_map * box.bb_array_mask

        corners_3d = box.corners()
        ''' x, y, z
              3-------2
             /|      /|
            7-------6 |
            | |     | |
            | 0-----|-1
            |/      |/
            4-------5
        '''

        # # Project 3D box corners to 2D
        corners_2d = view_points(corners_3d, camera_intrinsic, normalize=True)[:2, :]

        if corners_2d[0, 5] > corners_2d[0, 1]:
            box.expand_rotation = True # we see the left side of the object
            rotation = 'LEFT'
            
            left_face_indices = [0, 3, 7, 4]

            left_face_corners_2d = np.int32(corners_2d[:, left_face_indices].T)
            left_face_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(left_face_mask, left_face_corners_2d, True)

            # left plane function 
            x6, y6, z6 = corners_2d[0, 3], corners_2d[1, 3], corners_3d[2, 3]
            x4, y4, z4 = corners_2d[0, 4], corners_2d[1, 4], corners_3d[2, 4]
            x7, y7, z7 = corners_2d[0, 7], corners_2d[1, 7], corners_3d[2, 7]

            A = (y6 - y7) * (z4 - z7) - (z6 - z7) * (y4 - y7)
            B = (z6 - z7) * (x4 - x7) - (x6 - x7) * (z4 - z7)
            C = (x6 - x7) * (y4 - y7) - (y6 - y7) * (x4 - x7)
            D = - (A*x6 + B*y6 + C*z6)

            depth_map_box = depth_update(depth_map_box, left_face_mask, A, B, C, D)

        else:
            box.expand_rotation = False # we see the right side of the object
            rotation = "RIGHT"

            right_face_indices = [1, 2, 6, 5]

            right_face_corners_2d = np.int32(corners_2d[:, right_face_indices].T)
            right_face_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(right_face_mask, right_face_corners_2d, True)
            
            # right plane function 
            x6, y6, z6 = corners_2d[0, 6], corners_2d[1, 6], corners_3d[2, 6]
            x7, y7, z7 = corners_2d[0, 5], corners_2d[1, 5], corners_3d[2, 5]
            x4, y4, z4 = corners_2d[0, 1], corners_2d[1, 1], corners_3d[2, 1]

            A = (y6 - y7) * (z4 - z7) - (z6 - z7) * (y4 - y7)
            B = (z6 - z7) * (x4 - x7) - (x6 - x7) * (z4 - z7)
            C = (x6 - x7) * (y4 - y7) - (y6 - y7) * (x4 - x7)        
            D = - (A*x6 + B*y6 + C*z6)

            depth_map_box = depth_update(depth_map_box, right_face_mask, A, B, C, D)
        
        # print(f"Box {i}, Rotation: {rotation}, \tCategory: {box.name}, ")

               
        # Back plane function
        back_face_indices = [4, 5, 6, 7]

        back_face_corners_2d = np.int32(corners_2d[:, back_face_indices].T)
        back_face_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(back_face_mask, back_face_corners_2d, True)

        # back plane function 
        x6, y6, z6 = corners_2d[0, 6], corners_2d[1, 6], corners_3d[2, 6]
        x7, y7, z7 = corners_2d[0, 7], corners_2d[1, 7], corners_3d[2, 7]
        x4, y4, z4 = corners_2d[0, 4], corners_2d[1, 4], corners_3d[2, 4]

        A = (y6 - y7) * (z4 - z7) - (z6 - z7) * (y4 - y7)
        B = (z6 - z7) * (x4 - x7) - (x6 - x7) * (z4 - z7)
        C = (x6 - x7) * (y4 - y7) - (y6 - y7) * (x4 - x7)
        D = - (A*x6 + B*y6 + C*z6)

        depth_map_back = depth_update(depth_map_box, back_face_mask, A, B, C, D)

        depth_map_final[depth_map_final==0] = depth_map_back[depth_map_final==0]
        depth_map_final[depth_map_final==0] = depth_map_box[depth_map_final==0]

        if plot:
            fig, ax = plt.subplots(figsize=(16, 9))
            ax.axis('off')
            ax.imshow(image)
            non_zero_indices = np.nonzero(depth_map_final)
            non_zero_depths = depth_map_final[non_zero_indices]
            scatter = ax.scatter(non_zero_indices[1], non_zero_indices[0], c=non_zero_depths, cmap='jet', s=0.01)
            # vmin, vmax = np.min(non_zero_depths), np.max(non_zero_depths)
            scatter.set_norm(Normalize(vmin=vmin, vmax=vmax))
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(scatter, cax=cax)
            cbar.set_label('Depth (meters)')
            output_path = os.path.join(scene_dir, f"dok_{i}_{box.name}.png")
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            print(f'dok_{i}_{box.name}')

    cmap = plt.get_cmap('jet')
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    colored_depth_map = cmap(norm(depth_map_final))
    colored_depth_map_uint8 = (colored_depth_map[:, :, :3] * 255).astype(np.uint8)
    img_colored = Image.fromarray(colored_depth_map_uint8)
    output_path = os.path.join(scene_dir, f"depth_map.png")
    img_colored.save(output_path)


# def depth_update(depth_map, face, A, B, C, D, difference=15):
#     xs, ys = np.where(face == True)  # x --> rows, y --> columns

#     # left to right (row-wise)
#     for x, y in zip(xs + 1, ys):
#         if depth_map[x, y - 1] > 0:
#             new_value = (-A * x - B * y - D) / C
#             if new_value > 0 and (depth_map[x, y] == 0 or abs(depth_map[x, y] - new_value) > difference):
#                 depth_map[x, y] = new_value

#     # right to left (row-wise)
#     for x, y in zip(xs - 1, ys[::-1][1:]):
#         if y < depth_map.shape[1] - 1 and depth_map[x, y + 1] > 0:
#             new_value = (-A * x - B * y - D) / C
#             if new_value > 0 and (depth_map[x, y] == 0 or abs(depth_map[x, y] - new_value) > difference):
#                 depth_map[x, y] = new_value

#     # bottom to top (column-wise)
#     for y in np.unique(ys):
#         x_sorted = np.sort(xs[ys == y])[::-1]  # sort the lines from bottom to top
#         for ii in range(1, len(x_sorted)):
#             x = x_sorted[ii]
#             x_below = x_sorted[ii - 1]
#             if depth_map[x_below, y] > 0:
#                 new_value = (-A * x - B * y - D) / C
#                 if new_value > 0 and (depth_map[x, y] == 0 or abs(depth_map[x, y] - new_value) > difference):
#                     depth_map[x, y] = new_value

#     return depth_map

def depth_update(depth_map, face, A, B, C, D, std_factor=0.5):
    xs, ys = np.where(face == True)  # x --> rows, y --> columns

    # Calculate the mean and standard deviation of non-zero depth values
    non_zero_depths = depth_map[depth_map > 0]
    mean_depth = np.mean(non_zero_depths)

    median_depth = np.median(non_zero_depths)

    std_depth = np.std(non_zero_depths)

    # Define the range within which values are considered not outliers
    
    outlier_removal_depth = median_depth + std_depth
    max_valid_depth = mean_depth + std_depth
    # print(median_depth, mean_depth, std_depth, outlier_removal_depth)

    # left to right (row-wise)
    for x, y in zip(xs + 1, ys):
        if depth_map[x, y - 1] > 0:
            new_value = (- A* x - B*y - D) / C
            if 0 < new_value < max_valid_depth and (depth_map[x, y] > outlier_removal_depth or depth_map[x, y] == 0):
                depth_map[x, y] = new_value

    # right to left (row-wise)
    for x, y in zip(xs - 1, ys[::-1][1:]):
        if y < depth_map.shape[1] - 1 and depth_map[x, y + 1] > 0:
            new_value = (- A*x - B*y - D) / C
            if 0 < new_value < max_valid_depth and (depth_map[x, y] > outlier_removal_depth or depth_map[x, y] == 0):
                depth_map[x, y] = new_value

    # bottom to top (column-wise)
    for y in np.unique(ys):
        x_sorted = np.sort(xs[ys == y])[::-1]  # sort the lines from bottom to top
        for ii in range(1, len(x_sorted)):
            x = x_sorted[ii]
            x_below = x_sorted[ii - 1]
            if depth_map[x_below, y] > 0:
                new_value = (- A*x - B*y - D) / C
            if 0 < new_value < max_valid_depth and (depth_map[x, y] > outlier_removal_depth or depth_map[x, y] == 0):
                    depth_map[x, y] = new_value


    # min_valid_depth = mean_depth + 2*std_depth
    # max_valid_depth = mean_depth #+ 0.01*std_depth
    # print(median_depth, mean_depth, min_valid_depth)

    # # left to right (row-wise)
    # for x, y in zip(xs + 1, ys):
    #     if depth_map[x, y - 1] > 0:
    #         new_value = (- A* x - B*y - D) / C
    #         if new_value > 0 and (depth_map[x, y] >= min_valid_depth or depth_map[x, y] == 0):
    #             depth_map[x, y] = new_value

    # # right to left (row-wise)
    # for x, y in zip(xs - 1, ys[::-1][1:]):
    #     if y < depth_map.shape[1] - 1 and depth_map[x, y + 1] > 0:
    #         new_value = (- A*x - B*y - D) / C
    #         if new_value > 0 and (depth_map[x, y] >= min_valid_depth or depth_map[x, y] == 0):
    #             depth_map[x, y] = new_value

    # # bottom to top (column-wise)
    # for y in np.unique(ys):
    #     x_sorted = np.sort(xs[ys == y])[::-1]  # sort the lines from bottom to top
    #     for ii in range(1, len(x_sorted)):
    #         x = x_sorted[ii]
    #         x_below = x_sorted[ii - 1]
    #         if depth_map[x_below, y] > 0:
    #             new_value = (- A*x - B*y - D) / C
    #         if new_value > 0 and (depth_map[x, y] >= min_valid_depth or depth_map[x, y] == 0):
    #                 depth_map[x, y] = new_value

    return depth_map



# def depth_update(depth_map, face, A, B, C):
#     xs, ys = np.where(face == True) # x --> rows , y --> columns

#     # left to right (row-wise)
#     for x, y in zip(xs+1, ys):
#         if depth_map[x, y] == 0 and depth_map[x, y-1] > 0:
#             depth_map[x, y] = depth_map[x, y-1] + (A / C) + (B / C)

#     # right to left (row-wise)
#     for x, y in zip(xs-1, ys[::-1][1:]):
#         if y < depth_map.shape[1] - 1 and depth_map[x, y+1] == 0: 
#             continue
#         elif y < depth_map.shape[1] - 1 and depth_map[x, y] == 0:
#             depth_map[x, y] = depth_map[x, y+1] + (A / C) + (B / C)

#     # bottom to top (column-wise)
#     for y in np.unique(ys):
#         x_sorted = np.sort(xs[ys == y])[::-1] # sort the lines from bottom to top
#         for ii in range(1, len(x_sorted)):
#             x = x_sorted[ii]
#             x_below = x_sorted[ii - 1]
#             if depth_map[x, y] == 0:
#                 depth_map[x, y] = depth_map[x_below, y] + (B / C)

#     return depth_map


def depth_update_old(depth_map, face, A, B, C):
    xs, ys = np.where(face == True) # x --> rows , y --> columns

    # left to right (row-wise)
    for x, y in zip(xs+1, ys):
        if depth_map[x, y] == 0 and depth_map[x, y-1] > 0:
            depth_map[x, y] = - A/C + depth_map[x, y-1]

    # right to left (row-wise)
    for x, y in zip(xs-1, ys[::-1][1:]):
        if y < depth_map.shape[1] - 1 and depth_map[x, y+1] == 0: 
            continue
        elif y < depth_map.shape[1] - 1 and depth_map[x, y] == 0:
            depth_map[x, y] = - A/C + depth_map[x, y+1]

    # bottom to top (column-wise)
    for y in np.unique(ys):
        x_sorted = np.sort(xs[ys == y])[::-1] # sort the lines from bottom to top
        for ii in range(1, len(x_sorted)):
            x = x_sorted[ii]
            x_below = x_sorted[ii - 1]
            if depth_map[x, y] == 0:
                depth_map[x, y] = - B/C + depth_map[x_below, y]
    
    return depth_map

depth_path = r'/imec/users/bourch01/dl4ms/SelfSupervision_dok/labels/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402928112460_depth_map.png'
depth_map = np.array(Image.open(depth_path))
del depth_path

z_buffer_code(img, depth_map, filtered_boxes, cam_intrinsic, plot = True, scene_dir=base_output_dir, name='dok')
print('end')

print()

