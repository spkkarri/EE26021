import os
import shutil


source_dirs = {
    'porcelain_MPID': 0,
    'glass_MPID': 1,
    'composite_MPID': 2
}
target_root = 'yolo_dataset'
subsets = ['train', 'val'] 

for material, class_id in source_dirs.items():
    for subset in subsets:
        
        src_img_path = os.path.join(material, subset, 'images')
        src_lbl_path = os.path.join(material, subset, 'labels')
        
        dest_img_path = os.path.join(target_root, subset, 'images')
        dest_lbl_path = os.path.join(target_root, subset, 'labels')

        os.makedirs(dest_img_path, exist_ok=True)
        os.makedirs(dest_lbl_path, exist_ok=True)

        if not os.path.exists(src_img_path):
            continue

        
        for img_file in os.listdir(src_img_path):
            shutil.copy(os.path.join(src_img_path, img_file), os.path.join(dest_img_path, img_file))

        
        for lbl_file in os.listdir(src_lbl_path):
            with open(os.path.join(src_lbl_path, lbl_file), 'r') as f:
                lines = f.readlines()
            
            with open(os.path.join(dest_lbl_path, lbl_file), 'w') as f:
                for line in lines:
                    parts = line.split()
                    if len(parts) > 0:
                        
                        parts[0] = str(class_id)
                        f.write(" ".join(parts) + "\n")

print("Merging complete! Your dataset is ready in the 'yolo_dataset' folder.")
