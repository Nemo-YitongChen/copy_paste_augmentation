# Copy-Paste-Augmentation Internship Project

## Working files
Use "copy_paste_project_script.py" for commandline training and saving the evaluation results (see "requirement.txt" and "Setup.md" first before running the script)

Use "copy_paste_augmentation_project.ipynb" to work on Google Colab.

Use "copy_paste__project.ipynb" to work on Jupyter Notebook to visualise the generated image fast and proper.


## Introduction
This project is focused on implementing Copy-Paste-Augmentation for COCO dataset and using Mask R-CNN model from Detectron2 to evaluate the training outcomes from different datasets and the influence of the augmentation method.

It is inspired by the research paper: "Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation" and by this project, we want to dig deeper for how to maximise the utility of current available datasets and how effective and efficient is the copy-paste-augmentation methodology.

The introduction details and presentation link are included in the poster.

### Findings
Copy-Paste-Augmentation is a great methodology to enlarge and fully exploit existing datasets since it has the ability to generate "new" data (image) from existing images while introducing new examples for the model to be further trained.

However, it has its restrictions: 

1. while trained on the same-size datasets, copy-paste augmented datasets have no improvements over the original dataset.
2. It takes significant time to augment the original datasets.
3. The quality of the generated images is hard to assess.


### Discussion
1. It is still unsure that after fully trained on a given dataset, how much improvements can Copy-Paste-Augmentation methodology achieve and how many newly generated images are needed as the minimum for the improvements.
2. For different downstream feature detection and image segmentation tasks, the quality of the generated images might be an essential creteria to make the fine-tuned model suit the tasks better. However, the quality is yet to be enhanced.
3. There can be potential risks of overfitting, but the fine-tuned models haven't been used for other images which are not in the COCO datasets.

## Project Plan and Achievement:

![image](https://github.com/Nemo-YitongChen/copy_paste_augmentation-project/assets/63221079/b65c1ca9-594e-4e34-bf25-66833ef09cc9)


## Skills Acquired:
### 1. Implementation Ability: 
Utilised State-of-Art Machine Learning models, implemented augmentation methods, and built a reusable training pipeline.
Acknowledge![image](https://github.com/Nemo-YitongChen/copy_paste_augmentation-project/assets/63221079/b550853a-ba19-4c35-8d0f-bd8789bd9706)


   
### 2. Dataset Preparation: 
prepared datasets according to different training requirements.
![image](https://github.com/Nemo-YitongChen/copy_paste_augmentation-project/assets/63221079/9f1a797e-70ae-425d-bc89-f41b9d31e453)


### 3. Project delivery:
Delivered runnable source codes in Jupyter Notebook and Commandline Script.
![image](https://github.com/Nemo-YitongChen/copy_paste_augmentation-project/assets/63221079/4ef5a3ab-4dc0-4e88-b9ba-1c3a99c91fce)

### 4. Evaluation:
Used COCOEvaluator to evaluate the quality of fine-tuned models.
<img width="1053" alt="image" src="https://github.com/Nemo-YitongChen/copy_paste_augmentation-project/assets/63221079/e68fcba2-65e4-437e-b5a0-9e1410612754">

## Experience
This internship is a great opportunity for me where I have completed a Machine Learning project all by myself with the support of my supervisor.
In this experience, I've mainly done the following activities:
1. Acquired and enhanced knowledge base from relevant research papers
2. Explored COCO dataset annotation and customised the annotation formats to be used for Detectron2
3. Implemented reusable augmentation training pipeline using self-defined functions and pre-trained models from Detectron2
4. Implemented most Copy-Paste Variation methods: Transformaton, scaling, rotation, and flipping.
5. Trained and tested the fine-tuned models with original and different augmentated datasets using COCOEvaluator with Box AP metrics.
6. Summarised and discussed future improvement and collaboration of this project with other machine learning projects.


## Achievements
Copy-paste-augmentation generated images:
![image](https://github.com/Nemo-YitongChen/copy_paste_augmentation-project/assets/63221079/9a7d63fc-3f71-4bb0-8a22-51b6d62d0500)


![image](https://github.com/Nemo-YitongChen/copy_paste_augmentation-project/assets/63221079/3cc0ee28-1e10-40d6-bcf3-53233560737e)



![image](https://github.com/Nemo-YitongChen/copy_paste_augmentation-project/assets/63221079/c53cffd4-86ed-463a-b1fd-1601228c8b08)



