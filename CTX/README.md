## Explanations of files in 'CTX/' directory:

* RobbinsCraters_tab.txt: Robbins crater dataset that we used as ground-truth targets
* original_images/ : the directory stores the original CTX images of our investigation region
* resized_CTX/: the directory stores the resized CTX images
* cropped_images/: the directory stores the CTX tiles cropped from the panoramic view of our investigation region
* stacked_CTX.png: the panoramic view our investigation region after stacking the 9 CTX images with right order
* gt_stacked_CTX.png: the panoramic view with the ground-truth targets mapped from Robbins crater dataset
* detections.png: our visualized detection result
* predictions.txt: the detections result
* pred_results/: the directory stores the detection results from the images with different resolution.
            For example, 'predictions_1248.txt' stores the result detected on the images of 1248x1248 resolution.
            'target_size.txt' stores the craters listed in the Robbins dataset.
            From these result, we can generated the cumulative size-number distribution.
* selected_craters_XY.txt: the labels mapped from Robbins crater dataset, with x1y1x2y2 format.
* visualization.txt: the normalized labels mapped from Robbins crater dataset, with x1y1x2y2 format for easier visualization on images with any resolution.
