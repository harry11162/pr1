# ECE276A Project 1

## Pixel Classification
We use Naive Bayes Model to classify the pixels.

The steps to use this code is:
1. Run `generate_rgb_data.py` to compute the parameters for the Bayes model. The parameters are shown in standard output.
2. Copy and paste the parameters into `pixel_classifier.py`.
3. Run `test_pixel_classifier.py` to see the results.

## Bin Detection
We use a Bayes Model to classify the pixels and segment the image.
Then we compare the contours of the segmented image to the known contours.
The known contours are computed using the same Bayes Model on the training images, so that the two are unified for comparison.

The steps to use this code is:
1. Run `train_model.py` to compute the parameters for the Bayes model. The parameters are shown in standard output.
2. Copy and paste the parameters into `bin_detector.py`.
3. Run `train_contours.py` to use the Bayes model to get contours of the objects in training images. The contours will be saved to `known_contours.pkl`.
4. Run `test_bin_detector.py` to see the results.