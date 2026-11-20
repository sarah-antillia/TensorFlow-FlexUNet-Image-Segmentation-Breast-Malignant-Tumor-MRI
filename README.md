<h2>TensorFlow-FlexUNet-Image-Segmentation-Breast-Malignant-Tumor-MRI (2026/11/21)</h2>
<h3>Breast-Malignant-Tumor-MRI: AI Generated Pseudo Masks Segmentation Challenge</h3>
Toshiyuki Arai<br>
Software Laboratory antillia.com<br><br>
This is the first experiment of Image Segmentation for <b>Breast-Malignant-Tumor-MRI </b>
 based on
our <a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">TensorFlowFlexUNet Model</a>
 (<b>TensorFlow Flexible UNet Image Segmentation Model for Multiclass</b>) and a 512x512-pixel PNG
<a href="https://drive.google.com/file/d/1lgwvj7C3GNGIWHfhiEpxwIt1ME0zzzkx/view?usp=sharing">
<b>Breast-Malignant-Tumor-MRI-ImageMask-Datasaet.zip</b></a>
 (<a href="https://www.mit.edu/~amini/LICENSE.md">MIT</a>
) derived  by us from the Kaggle
<br><br>
<a href="https://www.kaggle.com/datasets/abenjelloun/breast-mri-tumor-classification-dataset">
<b>Breast MRI Tumor Classification Dataset
</b>
</a> by A.Benjelloun.
<br><br>
<hr>
<b>Actual Image Segmentation for Breast-Malignant-Tumor-MRI Images of 512x512 pixels</b><br>
As shown below, the inferred masks resemble the ground-truth masks. <br>
<br>
<br>
<table>
<tr>
<th width="320" height="auto">Input: image</th>
<th width="320" height="auto">Mask (ground_truth)</th>
<th width="320" height="auto">Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Malignant-Tumor-MRI/mini_test/images/BREASTDX-01-0003_11436.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Malignant-Tumor-MRI/mini_test/masks/BREASTDX-01-0003_11436.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Malignant-Tumor-MRI/mini_test_output/BREASTDX-01-0003_11436.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Malignant-Tumor-MRI/mini_test/images/BREASTDX-01-0026_16488.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Malignant-Tumor-MRI/mini_test/masks/BREASTDX-01-0026_16488.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Malignant-Tumor-MRI/mini_test_output/BREASTDX-01-0026_16488.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Malignant-Tumor-MRI/mini_test/images/BREASTDX-01-0065_2136.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Malignant-Tumor-MRI/mini_test/masks/BREASTDX-01-0065_2136.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Malignant-Tumor-MRI/mini_test_output/BREASTDX-01-0065_2136.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>1. Dataset Citation</h3>
The dataset used here was derived from the following Kaggle:<br><br>

<a href="https://www.kaggle.com/datasets/abenjelloun/breast-mri-tumor-classification-dataset">
<b>Breast MRI Tumor Classification Dataset
</b>
</a> <br>
<b>MRI-based dataset for classifying breast tumors into benign and malignant
</b>
<br> by A.Benjelloun.
<br><br>
The following explanation was taken from the website above. 
<br><br>
<b>About Dataset</b><br>
<b>Dataset Description</b><br>
This dataset contains breast MRI images curated from The Cancer Imaging Archive (TCIA) Breast 
Diagnosis collection source: https://www.cancerimagingarchive.net/collection/breast-diagnosis/.
<br><br>
Originally, the full collection included over 101,000 medical images across different modalities 
such as MRI, CT, PET, MG, along with clinical metadata including Diagnosis, Molecular Test, 
Demographic, Pathology Detail, Radiomic Feature, and Measurement. For the purpose of this study, 
we filtered and retained only the MRI images to focus specifically on breast tumor classification.
<br><br>
It includes two sets of images(each contains almost 25,000 MRI scans):
<ul>
<li>Preprocessed images resized to 224x224 pixels suitable for deep learning model training and experimentation.
</li>
<li>
Original resolution images preserving full detail for further analysis or custom preprocessing.
</li>
</ul>
The dataset was carefully cleaned and prepared following these steps:
<ul>
<li>
Filtering MRI series to retain only MR modality images from the original multi-modality collection.
</li>
<li>
Transposing and correcting metadata tables that were initially disorganized (rows and columns flipped).
</li>
<li>
Deriving a clear classification label (“Benign” or “Malignant”) for each patient using keyword-based mapping on the Pathology column, supplemented by manual review of clinical reports when data was missing or ambiguous.
</li>
<li>
Removing entries with missing values in critical metadata fields to ensure data quality and reliable labels.
</li>
<li>
Linking each MRI series to the corresponding patient-level diagnosis to create labeled samples.
</li>
</ul>
<b>Official TCIA Citation</b><br>
Bloch, B. Nicolas, Jain, Ashali, & Jaffe, C. Carl. (2015). BREAST-DIAGNOSIS [Data set]. <br>
The Cancer Imaging Archive. http://doi.org/10.7937/K9/TCIA.2015.SDNRQXXR
<br><br>
<b>License</b><br>
<a href="https://www.mit.edu/~amini/LICENSE.md">MIT</a>
<br>
<br>
<h3>
2. Breast-Malignant-Tumor-MRI ImageMask Dataset
</h3>
<h3>2.1 Download ImageMask Dataset</h3>
 If you would like to train this Breast-Malignant-Tumor-MRI Segmentation model,
 please download the dataset from the Google Drive 
<a href="https://drive.google.com/file/d/1lgwvj7C3GNGIWHfhiEpxwIt1ME0zzzkx/view?usp=sharing">
<b>Breast-Malignant-Tumor-MRI-ImageMask-Datasaet.zip</b>.
</a> 
Expand the downloaded and put it under the <b>./dataset</b> folder.
<br>
<pre>
./dataset
└─Breast-Malignant-Tumor-MRI
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
         ├─images
         └─masks
</pre>
<br>
<b>Breast-Malignant-Tumor-MRI Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/Breast-Malignant-Tumor-MRI/Breast-Malignant-Tumor-MRI_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images in the training and valid datasets is large enough to use for the
 training set of our segmentation model.
<br>
<h3>2.2 Derivation of ImageMask Dataset</h3>
The folder structure of the original breast_mri_dataset is as follows, but it contains no annotation (mask) files.
<pre>
./breast_mri_dataset
    ├─test
    │   ├─Benign
    │   └─Malignant
    ├─train
    │   ├─Benign
    │   └─Malignant
    └─val
         ├─Benign
         └─Malignant
</pre>
<b>Step 1</b><br>
We generated a 512x512-pixel PNG master images from the JPG files in 
<b>train/Malignant</b> subfolder.
<br><br>
<b>Step 2</b><br>
We generated the first pseudo masks corresponding to the master images
by applying a segmentation (inference) method of the pretrained model 
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-BreastDM-DCE-MRI">
TensorFlow-FlexUNet-Image-Segmentation-BreastDM-DCE-MRI
</a> to the master images.
<br><br>
<b>Step 3</b><br>
We generated an ImageMask dataset from all pairs of the first pseudo masks and the master images.
<br><br>
<b>Step 4</b><br>
We generated a new pretrained FlexUNet segmentation model by using the ImageMask 
dataset derived at Step 3.
<br><br>
<b>Step 5</b><br>
We generated the new pseudo masks corresponding to the master images by applying a segmentation method of the new 
pretrained FlexUNet model obtained at Step 4.
<br><br>
We iteratively repeated Step3 to 5 two times to generate our own ImageMaskDataset.
<br><br>
<h3>2.3 Train Sample Images and Masks</h3>
<b>Train_sample_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Breast-Malignant-Tumor-MRI/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_sample_masks</b><br>
<img src="./projects/TensorFlowFlexUNet/Breast-Malignant-Tumor-MRI/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<h3>
3. Train TensorFlowFlexUNet Model
</h3>
 We trained the Breast-Malignant-Tumor-MRI TensorFlowFlexUNet model using the following
<a href="./projects/TensorFlowFlexUNet/Breast-Malignant-Tumor-MRI/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/Breast-Malignant-Tumor-MRI and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
This runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>
<b>Model parameters</b><br>
Defined a small <b>base_filters=16 </b> and large <b>base_kernels=(11,11)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large <b>num_layers=8</b> (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
; You may specify your own UNet class derived from our TensorFlowFlexModel
model         = "TensorFlowFlexUNet"
generator     =  False
image_width    = 512
image_height   = 512
image_channels = 3
num_classes    = 2
base_filters   = 16
base_kernels   = (11,11)
num_layers     = 8
dropout_rate   = 0.05
dilation       = (1,1)
</pre>
<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>
<b>Loss and metrics functions</b><br>
Specified "categorical_focal_dice_loss" and <a href="./src/dice_coef_multiclass.py">"dice_coef_hybrid"</a>.<br>
<pre>
[model]
loss           = "categorical_focal_dice_loss"
metrics        = ["dice_coef_hybrid"]
hybrid_alpha   = 1.4
hybrid_beta    = 0.6
</pre>
<b>Dataset class</b><br>
Specified <a href="./src/ImageCategorizedMaskDataset.py">ImageCategorizedMaskDataset</a> class.<br>
<pre>
[dataset]
class_name    = "ImageCategorizedMaskDataset"
</pre>
<br>
<b>Learning rate reducer callback</b><br>
Enabled the learning_rate_reducer callback and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with the patience parameter.
<pre>
[train]
patience      = 10
</pre>
<b>RGB Color map</b><br>
Specified RGB color map dict for Breast-Malignant-Tumor-MRI 1+1 classes.<br>
<pre>
[mask]
mask_datatyoe    = "categorized"
mask_file_format = ".png"
;Breast-Malignant-Tumor-MRI RGB color map dict for 1+1 classes.
; Background: black,  Malignant-Tumor: red
rgb_map = {(0,0,0):0,(255,0,0):1}
</pre>
<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInferencer.py">epoch_change_infer callback</a></b>.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
num_infer_images         = 6
</pre>
By using this callback, on every epoch change, the inference procedure can be called
 for 6 images in the <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> 
<br> 
<!--
As shown below, early in the model training, the predicted masks from our FlexUNet segmentation model showed 
discouraging results.
 However, as training progressed through the epochs, the predictions gradually improved. 
 <br> 
<br>
-->
<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/Breast-Malignant-Tumor-MRI/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 19,20,21)</b><br>
<img src="./projects/TensorFlowFlexUNet/Breast-Malignant-Tumor-MRI/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (epoch 39,40,41)</b><br>
<img src="./projects/TensorFlowFlexUNet/Breast-Malignant-Tumor-MRI/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>
In this experiment, the training process was stopped at epoch 41 by EarlyStoppingCallback.<br>
<img src="./projects/TensorFlowFlexUNet/Breast-Malignant-Tumor-MRI/asset/train_console_output_at_epoch41.png" width="1024" height="auto"><br>
<br>
<a href="./projects/TensorFlowFlexUNet/Breast-Malignant-Tumor-MRI/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Breast-Malignant-Tumor-MRI/eval/train_metrics.png" width="520" height="auto"><br>
<br>
<a href="./projects/TensorFlowFlexUNet/Breast-Malignant-Tumor-MRI/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Breast-Malignant-Tumor-MRI/eval/train_losses.png" width="520" height="auto"><br>
<br>
<h3>
4. Evaluation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/Breast-Malignant-Tumor-MRI</b> folder,
and run the following bat file to evaluate the TensorFlowUNet model for Breast-Malignant-Tumor-MRI.<br>
<pre>
>./2.evaluate.bat
</pre>
This runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer.config
</pre>
Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/Breast-Malignant-Tumor-MRI/asset/evaluate_console_output_at_epoch41.png" width="1024" height="auto">
<br><br>Image-Segmentation-Breast-Malignant-Tumor-MRI

<a href="./projects/TensorFlowFlexUNet/Breast-Malignant-Tumor-MRI/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_focal_dice_loss) on this Breast-Malignant-Tumor-MRI/test was very low, 
but dice_coef_hybrid was not so high,
 as shown below.
<br>
<pre>
categorical_focal_dice_loss,0.0015
dice_coef_hybrid,0.9035
</pre>
<br>
<h3>
5. Inference
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/Breast-Malignant-Tumor-MRI</b> folder
and run the following bat file to infer segmentation regions for images using the trained TensorFlowUNet model for Breast-Malignant-Tumor-MRI.<br>
<pre>
>./3.infer.bat
</pre>
This runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Breast-Malignant-Tumor-MRI/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/Breast-Malignant-Tumor-MRI/asset/mini_test_masks.png" width="1024" height="auto"><br>
<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/Breast-Malignant-Tumor-MRI/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks for Breast-Malignant-Tumor-MRI Images of 512x512 pixels</b><br>
As shown below, the inferred masks look similar to the ground truth masks except for the third and 
sixth cases.<br>
<br><br>
<table>
<tr>
<th width="320" height="auto">Image</th>
<th width="320" height="auto">Mask (ground_truth)</th>
<th width="320" height="auto">Inferred-mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Malignant-Tumor-MRI/mini_test/images/BREASTDX-01-0018_8523.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Malignant-Tumor-MRI/mini_test/masks/BREASTDX-01-0018_8523.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Malignant-Tumor-MRI/mini_test_output/BREASTDX-01-0018_8523.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Malignant-Tumor-MRI/mini_test/images/BREASTDX-01-0022_6027.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Malignant-Tumor-MRI/mini_test/masks/BREASTDX-01-0022_6027.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Malignant-Tumor-MRI/mini_test_output/BREASTDX-01-0022_6027.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Malignant-Tumor-MRI/mini_test/images/BREASTDX-01-0034_7795.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Malignant-Tumor-MRI/mini_test/masks/BREASTDX-01-0034_7795.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Malignant-Tumor-MRI/mini_test_output/BREASTDX-01-0034_7795.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Malignant-Tumor-MRI/mini_test/images/BREASTDX-01-0068_15322.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Malignant-Tumor-MRI/mini_test/masks/BREASTDX-01-0068_15322.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Malignant-Tumor-MRI/mini_test_output/BREASTDX-01-0068_15322.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Malignant-Tumor-MRI/mini_test/images/BREASTDX-01-0070_9894.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Malignant-Tumor-MRI/mini_test/masks/BREASTDX-01-0070_9894.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Malignant-Tumor-MRI/mini_test_output/BREASTDX-01-0070_9894.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Malignant-Tumor-MRI/mini_test/images/BREASTDX-01-0046_9319.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Malignant-Tumor-MRI/mini_test/masks/BREASTDX-01-0046_9319.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Malignant-Tumor-MRI/mini_test_output/BREASTDX-01-0046_9319.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
References
</h3>
<b>1. BreastDM: A DCE-MRI dataset for breast tumor image segmentation and classification</b><br>
Xiaoming Zhao, Yuehui Liao, Jiahao Xie, Xiaxia He, Shiqing Zhang, Guoyu Wang <br>, Jiangxiong Fang
, Hongsheng Lu, Jun Yu <br>
<a href="https://doi.org/10.1016/j.compbiomed.2023.107255">https://doi.org/10.1016/j.compbiomed.2023.107255</a><br>
<a href="https://www.sciencedirect.com/science/article/abs/pii/S0010482523007205">
https://www.sciencedirect.com/science/article/abs/pii/S0010482523007205</a>
<br>
<br>
<b>2. Tensorflow-Image-Segmentation-Malignant-BreastDM</b><br>
Toshiyuki Arai <br>
<a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Malignant-BreastDM">
https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Malignant-BreastDM
</a>
<br>
<br>
<b>3. TensorFlow-FlexUNet-Image-Segmentation-BreastDM-DCE-MRI</b><br>
Toshiyuki Arai<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-BreastDM-DCE-MRI">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-BreastDM-DCE-MRI
</a>
<br><br>
<b>4. TensrFlow-FlexUNet-Image-Segmentation-Model</b><br>
Toshiyuki Arai<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model
</a>
<br><br>
