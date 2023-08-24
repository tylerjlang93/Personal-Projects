# AIplus-Practicum-Final
Final deliverable package for AI+ Practicum Team 1 - Summer 2023


To use the notebooks provided, data must be moved to the Data folder provided. Please see individual notebooks for any requirements regarding location references or structure of data, depending on if the data being presented is being used to fine-tune a model, or use a pre-trained model to predict on a set of data. 

All of our prediction methods require pre-processing of the data, the processes of which can found in "data_preprocessing." Different methods of pre-processing are applicable to their specific dataset, such as the notebook "data_preprocess_BUSI," which is applicable to the public BUSI dataset, and the OASBUD notebook for the public OASBUD dataset. "Image Preprocessing experiments" is a notebook with various experimentations we performed to arrive at our final pre-processing strategy. 

<<Brian can you please give more information on what pre-processing notebooks are required to run for the 1-step pipeline?>>

"AIplus_1step_pipeline.ipynb" is our current, best-performing notebook for predicting new images, and will all steps associated with object detection, including providing the coordinates to crop the tumor from the image, and predict the type of tumor as benign or malignant. Please see notebook-specific instructions for help on running the notebook. For the notebook to successfully run, .yaml files are included to guide the YOLO model, as needed by its usage requirements.

<<Which of the yolo notebooks are required? Can we move any of that into Deprecated?>>


Within the "Deprecated" folder, notebooks can be found regarding processess that we experimented with, and were ultimately outperformed by our current model architecture via the 1-step YOLO object detection model. Included processess include a notebook dedicated to post-cropped tumor classification (benign or malignant) using a variety of deep learning architectures and strategies, and is used to ultimately train a DeepNet model that, when provided croppped tumor images, can achieve accuracy of 85% on an unseen test set. It also includes our Faster-RCNN model previously used for object detection, which was then replaced by YOLO, but is mentioned in our final report. "aiplus_baseline" contains various experiments ran on different formats of the YOLO model, to find the best model to use in the 1-step pipeline. 

A formal report of findings can be found in "Final Report.pdf"

----------------------------------------------------------------------