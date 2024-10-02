from fusionnet_main import train

''' CODE TO DEBUG, DELETE IT AT THE END'''


train_image_path = f"../expand_jbf/paths/nuscenes_train_image.txt"
train_depth_path = "../expand_jbf/paths/nuscenes_train_depth_map.txt"
train_response_path = "../expand_jbf/paths/nuscenes_train_confidence_map.txt"
train_ground_truth_path = "../expand_jbf/paths/nuscenes_train_lidar.txt"

val_image_path = "../expand_jbf/paths/nuscenes_validation_image.txt"
val_depth_path = "../expand_jbf/paths/nuscenes_validation_depth_map.txt"
val_response_path = "../expand_jbf/paths/nuscenes_validation_confidence_map.txt"
val_ground_truth_path = "../expand_jbf/paths/nuscenes_validation_lidar.txt"

batch_size = 1
n_height = 224
n_width = 224

input_channels_image = 3
input_channels_depth = 2
normalized_image_range = [0, 1]

encoder_type = ["fusionnet18", "batch_norm"]
n_filters_encoder_image = [32, 64, 128, 256, 256, 256]
n_filters_encoder_depth = [16, 32,  64, 128, 128, 128]
fusion_type = "weight_and_project"
decoder_type = ["multiscale", "batch_norm"]
n_filters_decoder = [256, 256, 128, 64, 64, 32]
n_resolutions_decoder = 1
min_predict_depth = 1.0
max_predict_depth = 100.0

weight_initializer = "kaiming_uniform"
activation_func = "leaky_relu"

learning_rates = [1e-3]
learning_schedule = [450]

loss_func = "l1"
w_smoothness = 0.0
w_lidar_loss = 2.0
w_weight_decay = 0.0
loss_smoothness_kernel_size = -1
outlier_removal_kernel_size = 7
outlier_removal_threshold = 1.5
ground_truth_dilation_kernel_size = -1

augmentation_probabilities = [1.00]
augmentation_schedule = [-1]
augmentation_random_crop_type = ['horizontal', 'vertical']
augmentation_random_brightness = [0.80, 1.20]
augmentation_random_contrast = [0.80, 1.20]
augmentation_random_saturation = [0.80, 1.20]
augmentation_random_flip_type = "horizontal"

min_evaluate_depth = 0
max_evaluate_depth = 100

# checkpoint_dirpath = "trained_fusionnet/fus18project6ms1bn_16x448x448_lr0-1e3_100_aug0-100_100_bri080-120_con080-120_sat080-120_hflip_l1_sm000_wd000_outrm7-150_dilate0_min1max100_lidar_loss200_interp_with_reproj"
checkpoint_dirpath = f'/media/bourcha/F6C0FAB4C0FA79E7/bourcha/trained_fusionnet/fus18project6ms1bn_16x448x448_lr0-1e3_100_aug0-100_100_bri080-120_con080-120_sat080-120_hflip_l1_sm000_wd000_outrm7-150_dilate0_min1max100_lidar_loss200_interp_with_reproj'
# checkpoint_dirpath = checkpoint_dirpath + '/model-20000.pth'

n_step_per_checkpoint = 5000
n_step_per_summary = 5000
start_step_validation = 25000
n_thread = 0

restore_path = f'/media/bourcha/F6C0FAB4C0FA79E7/bourcha/trained_fusionnet/fus18project6ms1bn_16x448x448_lr0-1e3_100_aug0-100_100_bri080-120_con080-120_sat080-120_hflip_l1_sm000_wd000_outrm7-150_dilate0_min1max100_lidar_loss200_interp_with_reproj/model-20000.pth'


if __name__ == '__main__':

    # Training settings
    assert len(learning_rates) == len(learning_schedule)

    train(train_image_path=train_image_path,
          train_depth_path=train_depth_path,
          train_response_path=train_response_path,
          train_ground_truth_path=train_ground_truth_path,
          #validation inputs
          val_image_path=val_image_path,
          val_depth_path=val_depth_path,
          val_response_path=val_response_path,
          val_ground_truth_path=val_ground_truth_path,
          # Batch settings
          batch_size=batch_size,
          n_height=n_height,
          n_width=n_width,
          # Input settings
          input_channels_image=input_channels_image,
          input_channels_depth=input_channels_depth,
          normalized_image_range=normalized_image_range,
          # Network settings
          encoder_type=encoder_type,
          n_filters_encoder_image=n_filters_encoder_image,
          n_filters_encoder_depth=n_filters_encoder_depth,
          fusion_type=fusion_type,
          decoder_type=decoder_type,
          n_filters_decoder=n_filters_decoder,
          n_resolutions_decoder=n_resolutions_decoder,
          min_predict_depth=min_predict_depth,
          max_predict_depth=max_predict_depth,
          # Weight settings
          weight_initializer=weight_initializer,
          activation_func=activation_func,
          # Training settings
          learning_rates=learning_rates,
          learning_schedule=learning_schedule,
          augmentation_probabilities=augmentation_probabilities,
          augmentation_schedule=augmentation_schedule,
          augmentation_random_crop_type=augmentation_random_crop_type,
          augmentation_random_brightness=augmentation_random_brightness,
          augmentation_random_contrast=augmentation_random_contrast,
          augmentation_random_saturation=augmentation_random_saturation,
          augmentation_random_flip_type=augmentation_random_flip_type,
          # Loss function settings
          loss_func=loss_func,
          w_smoothness=w_smoothness,
          w_weight_decay=w_weight_decay,
          loss_smoothness_kernel_size=loss_smoothness_kernel_size,
          w_lidar_loss=w_lidar_loss,
          ground_truth_outlier_removal_kernel_size=outlier_removal_kernel_size,
          ground_truth_outlier_removal_threshold=outlier_removal_threshold,
          ground_truth_dilation_kernel_size=ground_truth_dilation_kernel_size,
          # Evaluation settings
          min_evaluate_depth=min_evaluate_depth,
          max_evaluate_depth=max_evaluate_depth,
          # Checkpoint settings
          checkpoint_dirpath=checkpoint_dirpath,
          n_step_per_checkpoint=n_step_per_checkpoint,
          n_step_per_summary=n_step_per_summary,
          start_step_validation=start_step_validation,
          restore_path=restore_path,
          # Hardware settings
          device='cuda',
          n_thread=0)