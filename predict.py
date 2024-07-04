import os
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2

# Define input and output directories
input_dir = r"Sample Images"  # folder containing input images
output_dir = r"save"  # folder to save output images

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Set up configuration
cfg = get_cfg()
cfg.merge_from_file(r"config_mask_rcnn.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = r"model_final_Mask_RCNN.pth"
cfg.MODEL.DEVICE = "cpu"

predictor = DefaultPredictor(cfg)

# Loop through each image in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Add other extensions if needed
        # Read image
        im = cv2.imread(os.path.join(input_dir, filename))
        
        # Check if image was loaded correctly
        if im is None:
            print(f"Error: Could not load image {filename}. Skipping.")
            continue

        # Perform prediction
        outputs = predictor(im)

        # Visualize results
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        # Convert to a format suitable for saving
        out_image = out.get_image()[:, :, ::-1]

        # Save output image
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, out_image)

print("Processing completed.")
