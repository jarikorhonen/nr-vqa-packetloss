# nr-vqa-packetloss
No-Reference method for assessing visibility of packet loss artifacts

This is implementation of a distortion specific No-Reference Video Quality Metric (NR-VQM) 
for detecting visibility of packet loss artifacts. Unlike most of the other NR-VQMs known 
in the prior art, the proposed scheme operates at the frame level, not the sequence level, 
and therefore it can be used to estimate the position of the impacted frames, rather than 
assessing the sequence level quality only. The metric is based on hand-crafted features and 
conventional learning-based regression. Since the proposed metric uses distortion-specific 
features, it is computationally less complex than most general purpose video and image quality 
metrics.

The following files are included:

compute_features.m
    
    Use this Matlab function to compute FR and NR features for a video sequence and write
    them in a CSV file for further processing.
    
EPFL_PoliMi_4CIF_example.m

    This Matlab script shows an example how to use compute_features.m. For using the script,
    EPFL-PoliMi 4CIF video sequences need to be downloaded and decoded. You can download the
    database provided by the authors [here](http://vqa.como.polimi.it/).
    
train_and_validate_EPFL-PoliMi_4CIF.py

    This Python script shows an example how to train and validate a regression model to 
    predict frame and sequence level quality scores, using the features computed with
    EPFL_PoliMi_4CIF_example.m. The script implements "leave-one-out" validation, without
    using the additional training contents.

More details about the method will appear in the following publication:

J. Korhonen, “Learning-based prediction of packet loss artifact visibility in networked video,” 
*IEEE International Conference on Quality of Multimedia Experience (QoMEX’18)*, Sardinia, Italy, 
May 2018. (Accepted for publication.)
