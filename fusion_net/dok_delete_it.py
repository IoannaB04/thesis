import os

def check_file_exists(file_path):
    if os.path.isfile(file_path):
        print(f"File exists at: {file_path}")
    else:
        print(f"File does not exist at: {file_path}")

# Example usage
checkpoint_dirpath = f'/media/bourcha/F6C0FAB4C0FA79E7/bourcha/trained_fusionnet/fus18project6ms1bn_16x448x448_lr0-1e3_100_aug0-100_100_bri080-120_con080-120_sat080-120_hflip_l1_sm000_wd000_outrm7-150_dilate0_min1max100_lidar_loss200_interp_with_reproj'
checkpoint_dirpath = checkpoint_dirpath + '/model-20000.pth'
check_file_exists(checkpoint_dirpath)
