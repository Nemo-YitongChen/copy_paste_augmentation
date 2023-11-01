# Copy-Paste-Augmentation Internship Project

## Working files
Use "copy_paste_project_script.py" for command line training and saving the evaluation results (see "requirement.txt" and "Setup.md" first before running the script)

Use "copy_paste_augmentation_project.ipynb" to work on Google Colab.

Use "copy_paste__project.ipynb" to work on Jupyter Notebook to visualise the generated image quickly and properly.


## Introduction
This project is focused on implementing Copy-Paste-Augmentation for the COCO dataset and using the Mask R-CNN model from Detectron2 to evaluate the training outcomes from different datasets and the influence of the augmentation method.

It is inspired by the research paper: "Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation" and by this project, we want to dig deeper into how to maximise the utility of current available datasets and how effective and efficient the copy-paste-augmentation methodology is.

The introduction details and presentation link are included in the poster.

### Findings
Copy-Paste-Augmentation is a great methodology to enlarge and fully exploit existing datasets since it has the ability to generate "new" data (image) from existing images while introducing new examples for the model to be further trained.

However, it has its restrictions: 

1. while trained on the same-size datasets, copy-paste augmented datasets have no improvements over the original dataset.
2. It takes significant time to augment the original datasets.
3. The quality of the generated images is hard to assess.


### Discussion
1. It is still unsure after fully training on a given dataset, how many improvements can Copy-Paste-Augmentation methodology achieve and how many newly generated images are needed as the minimum for the improvements.
2. For different downstream feature detection and image segmentation tasks, the quality of the generated images might be an essential criterion to make the fine-tuned model suit the tasks better. However, the quality is yet to be enhanced.
3. There can be potential risks of overfitting, but the fine-tuned models haven't been used for other images which are not in the COCO datasets.

## Project Plan and Achievement:

![image](https://github.com/Nemo-YitongChen/copy_paste_augmentation-project/assets/63221079/b65c1ca9-594e-4e34-bf25-66833ef09cc9)


## Skills Acquired:
### 1. Implementation Ability: 
Utilised State-of-Art Machine Learning models, implemented augmentation methods, and built a reusable training pipeline.
<img width="900" alt="image" src="https://github.com/Nemo-YitongChen/copy_paste_augmentation-project/assets/63221079/38f40e70-7f74-40a4-98d5-245a55b363db">
   
### 2. Dataset Preparation: 
prepared datasets according to different training requirements.
![image](https://github.com/Nemo-YitongChen/copy_paste_augmentation-project/assets/63221079/9f1a797e-70ae-425d-bc89-f41b9d31e453)

### 3. Project delivery:
Delivered runnable source codes in Jupyter Notebook and Commandline Script.
![image](https://github.com/Nemo-YitongChen/copy_paste_augmentation-project/assets/63221079/4ef5a3ab-4dc0-4e88-b9ba-1c3a99c91fce)

### 4. Evaluation:
Used COCOEvaluator to evaluate the quality of fine-tuned models.
<img width="600" alt="image" src="https://github.com/Nemo-YitongChen/copy_paste_augmentation-project/assets/63221079/e68fcba2-65e4-437e-b5a0-9e1410612754">

## Experience
This internship is a great opportunity for me where I have completed a Machine Learning project all by myself with the support of my supervisor.
In this experience, I've mainly done the following activities:
1. Acquired and enhanced knowledge base from relevant research papers
2. Explored COCO dataset annotation and customised the annotation formats to be used for Detectron2
3. Implemented reusable augmentation training pipeline using self-defined functions and pre-trained models from Detectron2
4. Implemented most Copy-Paste Variation methods: transformation, scaling, rotation, and flipping.
5. Trained and tested the fine-tuned models with original and different augmented datasets using COCOEvaluator with Box AP metrics.
6. Summarised and discussed future improvement and collaboration of this project with other machine learning projects.


## Achievements
Copy-paste-augmentation generated images:
![image](https://github.com/Nemo-YitongChen/copy_paste_augmentation-project/assets/63221079/9a7d63fc-3f71-4bb0-8a22-51b6d62d0500)


![image](https://github.com/Nemo-YitongChen/copy_paste_augmentation-project/assets/63221079/340bc452-d736-41ce-8d02-36ba2bc89398)




![image](https://github.com/Nemo-YitongChen/copy_paste_augmentation-project/assets/63221079/c53cffd4-86ed-463a-b1fd-1601228c8b08)

## Reference List

Geiger A, Lenz P, Stiller C and Urtasun (2013) ‘R Vision meets robotics: The KITTI dataset’, The International Journal of Robotics Research, 32(11):1231-1237, doi:10.1177/0278364913491297


Ghiasi G, Cui Y, Srinivas A, Qian R, Tsung-Yi L, Cubuk E, Le Q and Zoph B (23 June 2021) ‘Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation’, 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Online, doi: DOI:10.1109/CVPR46437.2021.00294 


He K, Gkioxari G, Dollár P and Girshick R (2017) ‘Mask R-CNN’, 2017 IEEE International Conference on Computer Vision (ICCV), Venice, Italy, doi: 10.1109/ICCV.2017.322


Ren S, He K, Girshick R and Sun J (2016) ‘Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks’, IEEE Transactions on Pattern Analysis & Machine Intelligence, 39(6): 1137-1149, doi: 10.1109/TPAMI.2016.2577031



