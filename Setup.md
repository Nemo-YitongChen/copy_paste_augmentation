1. Install all required packages specified in the 'requirement.txt':
    Common missing packages: 'pyclipper', 'shapely'.
	pip install pyclipper
	pip install shapely
2. Setup the directories: 
    - PATH_TO_TRAIN_IMAGE
        - [all training images, from COCO training dataset: http://images.cocodataset.org/zips/train2017.zip]
        - annotations
            - instances_train2017.json
    - PATH_TO_TEST_IMAGE
        - [all testing images, from COCO validation dataset: http://images.cocodataset.org/zips/val2017.zip]
        - annotations
            - instances_val2017.json

3. Run the script with the following command:
	python3 copy_paste_project_script.py PATH_TO_TRAIN_IMAGE PATH_TO_TEST_IMAGE PATH_TO_OUTPUT_DIR ("test")

