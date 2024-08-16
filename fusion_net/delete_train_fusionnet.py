import argparse
from fusionnet_main import train

from config.config import get_cfg, get_parser

if __name__ == '__main__':

    args = get_parser().parse_args()
    cfg = get_cfg(args)

    # Training settings
    assert len(args.learning_rates) == len(args.learning_schedule)

    train(train_image_path=args.train_image_path,
          train_depth_path=args.train_depth_path,
          train_response_path=args.train_response_path,
          train_ground_truth_path=args.train_ground_truth_path,
          train_lidar_map_path=args.train_lidar_map_path,
          val_image_path=args.val_image_path,
          val_depth_path=args.val_depth_path,
          val_response_path=args.val_response_path,
          val_ground_truth_path=args.val_ground_truth_path,

          # Batch settings
          batch_size=cfg.MODEL.DEPTH.BATCH_SIZE,
          n_height=cfg.MODEL.DEPTH.HEIGHT,
          n_width=cfg.MODEL.DEPTH.WIDTH ,
          # Input settings
          input_channels_image=cfg.MODEL.DEPTH.INPUT_CHANNELS_IMAGE,
          input_channels_depth=cfg.MODEL.DEPTH.INPUT_CHANNELS_DEPTH,
          normalized_image_range=cfg.MODEL.DEPTH.NORMALIZE_IMAGE_RANGE,
          # Network settings
          encoder_type=cfg.MODEL.DEPTH.ENCODER.TYPE,
          n_filters_encoder_image=cfg.MODEL.DEPTH.ENCODER.FILTERS_IMAGE,
          n_filters_encoder_depth=cfg.MODEL.DEPTH.ENCODER.FILTERS_DEPTH ,
          fusion_type=cfg.MODEL.DEPTH.FUSION_TYPE,
          decoder_type=cfg.MODEL.DEPTH.DECODER.TYPE,
          n_filters_decoder=cfg.MODEL.DEPTH.DECODER.FILTERS,
          n_resolutions_decoder=cfg.MODEL.DEPTH.DECODER.RESOLUTION,
          min_predict_depth=cfg.MODEL.DEPTH.MIN_DEPTH,
          max_predict_depth=cfg.MODEL.DEPTH.MAX_DEPTH,
          # Weight settings
          weight_initializer=cfg.MODEL.DEPTH.WEIGHT_INITIALIZER,
          activation_func=cfg.MODEL.DEPTH.ACTIVATION_FUNCTION,
          # Training settings
          learning_rates=cfg.MODEL.DEPTH.LEARNING_RATE,
          learning_schedule=cfg.MODEL.DEPTH.LEARNING_SCEDULE,
          # Augmentation settings
          augmentation_probabilities=cfg.MODEL.DEPTH.AUGMENTATION.PROBABILITIES,
          augmentation_schedule=cfg.MODEL.DEPTH.AUGMENTATION.SCEDULE,
          augmentation_random_crop_type=cfg.MODEL.DEPTH.AUGMENTATION.CROP_TYPE ,
          augmentation_random_brightness=cfg.MODEL.DEPTH.AUGMENTATION.BRIGHTNESS,
          augmentation_random_contrast=cfg.MODEL.DEPTH.AUGMENTATION.CONTRAST,
          augmentation_random_saturation=cfg.MODEL.DEPTH.AUGMENTATION.SATURATION,
          augmentation_random_flip_type=cfg.MODEL.DEPTH.AUGMENTATION.FLIP_TYPE,
          # Loss function settings
          loss_func=cfg.MODEL.DEPTH.LOSS_FUNCTION,
          w_smoothness=args.w_smoothness,
          w_weight_decay=args.w_weight_decay,
          loss_smoothness_kernel_size=args.loss_smoothness_kernel_size,
          w_lidar_loss=args.w_lidar_loss,
          ground_truth_outlier_removal_kernel_size=args.outlier_removal_kernel_size,
          ground_truth_outlier_removal_threshold=args.outlier_removal_threshold,
          ground_truth_dilation_kernel_size=args.ground_truth_dilation_kernel_size,
          # Evaluation settings
          min_evaluate_depth=cfg.MODEL.DEPTH.MIN_DEPTH_EVALUATION,
          max_evaluate_depth=cfg.MODEL.DEPTH.MAX_DEPTH_EVALUATION,
          # Checkpoint settings
          checkpoint_dirpath=args.checkpoint_dirpath,
          n_step_per_checkpoint=args.n_step_per_checkpoint,
          n_step_per_summary=args.n_step_per_summary,
          start_step_validation=args.start_step_validation,
          restore_path=args.restore_path,
          # Hardware settings
          device='cuda',
          n_thread=10  # Number of threads for fetching
          )   
