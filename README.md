# End-to-End-Pipeline-for-Robotics-ML-Training-with-Synthetic-Data-on-Nvidia-Isaac-SIM

## Describing the different files used in the process:

### generate_data.sh: 
- What it is?: a script that automate the utilization of Isaac SIM to generate synthetic data
- What is does?: This script calls the data generation script "standalone_palletjack_sdg.py" and passes parameters to it. Because we want to create 3 different types of distractor objects, we need to run the data generation script 3 times, each time, passing different parameters. This scrip automates the process of running the data generation script 3 times. 
- It saves the synthetic images generated in the folders saved in the "OUTPUT_..." variables.

### standalone_palletjack_sdg.py
- What it is?: This is the script that actually generates the synthetic data (images). 
- What it does:?
   1. Opens a warehouse scene in NVIDIA Isaac Sim.
   2. Spawns a few pallet jack assets (the object we want to detect).
   3. Spawns distractor props (extra objects) to make scenes varied.
   4. Randomizes camera, lighting, object poses, and materials (domain randomization).
   5. Captures images + labels using a KITTI-format "writer" to build a training dataset.
   6. In a nutshell, It loads a warehouse 3D environment from Nvidia, fetches 3D assets from the Nvidia cloud database (both objects of interest and distractors), then uses the Replicator function to randomize the position and color of these objects in the scene. 

### local_train.ipynb
- What it is?: A Python script on Jupyter Notebook containing the Machine Learning Model.
- What it does?: Takes the synthetic images, trains a ML model on them and tests it to try to find the object of interest and then measures its precision. 


## STEP BY STEP

### 1. Generate Synthetic Data

From COURSE: Synthetic Data Generation for Perception Model Training in Isaac Sim
https://learn.nvidia.com/courses/course?course_id=course-v1:DLI+S-OV-30+V1&unit=block-v1:DLI+S-OV-30+V1+type@vertical+block@7fecaf9f66204c0ea35402fca5ae1b25
Generating a Synthetic Dataset Using Replicator > Activity: Understanding Basics of the SDG Script and Activity: Running the Script for Generating Training Data

#### 1.2. Clone repo: 
- git clone https://github.com/NVIDIA-AI-IOT/synthetic_data_generation_training_workflow.git
- Make sure you take note from where you saved the repo in your local machine because all other steps will need to be done from or to this folder (eg. running the scripts, saving synthetic images, running the ML model).

#### 1.3. Adjust generate_data.sh to clear cache
- Open generate_data.sh file with text editor
- add "--clear-cache --clear-data" to the end of the parameters list passed for each run. eg: "cmd //c "C:\isaacsim\python.bat" $SCRIPT_PATH --height 544 --width 960 --num_frames 2000 --distractors warehouse --data_dir $OUTPUT_WAREHOUSE --clear-cache --clear-data"
- This is to make sure after each run the cache is deleted to prevent the GPU from running out of memory

#### 1.4. Manually Delete Cache Files
- If previous runs have been done before, delete cache files from "C:...[YOUR LOCAL FOLDERS]...AppData\Local\ov\cache"
(Replace [YOUR LOCAL FOLDERS] with your specific local path to where you saved the github repository)
- This is to make sure after each run the cache is deleted to prevent the GPU from running out of memory

#### 1.5. Run generate_data.sh
- Locate and Configure generate_data.sh
- Go to my path: "C:...[YOUR LOCAL FOLDERS]...GitHub\synthetic_data_generation_training_workflow\local". Open the file "generate_data.sh" with a text editor. 
- Open file and insert the path where I saved Isaac SIM on my computer: eg: "C:\isaacsim"
- Check the path assigned to the "output variables", this is where the images will be saved

1.3.2. Run the script
- Go to the folder and double click on the file: "generate_data.sh"
- it will open Isaac SIM and start generating synthetic data image files to the output folders

#### 1.6. Check synthetic data generated
- Go to path assigned to the output variables open rgb files. 
- Their quantity should correspond to the number passed in the --num_frames parameters defined in generate_data.sh

_____________________________________________________________________________


### 2: Train and Test the Machine Learning Model with the generated Synthetic Data 

From COURSE: "Fine-Tuning and Validating an AI Perception Model" > "Lecture: Training a Model With Synthetic Data"
https://learn.nvidia.com/courses/course?course_id=course-v1:DLI+S-OV-30+V1&unit=block-v1:DLI+S-OV-30+V1+type@vertical+block@aced7cf26b974581baf48fae53b70341

#### 2.0 - Make sure you have completed all the previous steps to generate synthetic data: 
- Clone repo: https://github.com/NVIDIA-AI-IOT/synthetic_data_generation_training_workflow.git
- Configure generate_data.sh
- Run generate_data.sh
##### DON NOT CLONE THE REPO FROM: 
https://learn.nvidia.com/courses/course?course_id=course-v1:DLI+S-OV-30+V1&unit=block-v1:DLI+S-OV-30+V1+type@vertical+block@aced7cf26b974581baf48fae53b70341 :  “Fine-Tuning and Validating an AI Perception Model > Lecture: Training a Model With Synthetic Data”
```
Optional: Training Your Own Model
For those interested in training their own model, follow these steps using the Synthetic Data Generation Training Workflow:
1. Clone the GitHub project and navigate to the local_train.ipynb notebook.
```
- This repo does NOT contain important folders such as “/workspace/tao-experiments/palletjack_sdg/palletjack_data/distractors_warehouse/Camera/rgb” that the model users to fetch the data to train from. If you try running the model from this repo it will throw you an error.
- Instead, go to the folder where you saved the repo from the previous step ("Generate Synthetic Data") and run local_train.ipynb from there”. This repo DOES contain “/workspace/tao-experiments/palletjack_sdg/palletjack_data/distractors_warehouse/Camera/rgb” and the other needed folders because they get generated during the synthetic data generation step. (This is not in Nvidia's original documentation)

#### 2.1- Install Ubuntu and Open Ubuntu CLI
- Some of the code in the jupyter notebook that runs the model is for Linux, so if you are running from a windows machine you need to install Ubuntu (Linux environment for Windows) and run everything from the Ubuntu CLI 

#### 2.2- create a separate Conda environment that users python version 3.10: 
- "conda create -n tao-py310 python=3.10 
- If this environment has already been previously created, skip this step.

#### 2.3- Activate python 3.10 conda env: 
- "conda activate tao-py310"

#### 2.4- Connect to Nvidia's docker container:
- The jupyter notebook that runs this model needs to be run from a Docker container that has all libraries already installed
- Install Docker Desktop
- Open docker desktop application and click on the play button on the container in the container list
- in the ubuntu cli run "docker login nvcr.io"
- login to the Nvidia container (if already logged user and password were already saved, if not get user (API key) from https://org.ngc.nvidia.com/setup/api-keys and rotate password to get a new password)
- run "docker ps -a" to check active containers.

#### 2.5- Navigate to the mounted folder in Ubuntu with the jupyter notebook
- navigate, in Ubuntu CLI, to folder where the GitHub project for synthetic data generation was cloned (from project https://learn.nvidia.com/courses/course?course_id=course-v1:DLI+S-OV-30+V1&unit=block-v1:DLI+S-OV-30+V1+type@vertical+block@7fecaf9f66204c0ea35402fca5ae1b25: "Generating a Synthetic Dataset Using Replicator > Activity: Understanding Basics of the SDG Script": 
"cd /mnt/c/...[YOUR LOCAL FOLDERS]...GitHub/synthetic_data_generation_training_workflow/local"

#### 2.6- Open the notebook in this folder from Ubuntu CLI: 
- "jupyter notebook local_train.ipynb --allow-root"
- Copy the URL provided in my web browser, click on the notebook to open

- inside the notebook: DO NOT replace "# os.environ["LOCAL_PROJECT_DIR"] = "<LOCAL_PATH_OF_CLONED_REPO>" this line of code doesn't do anything since the path is automatically fetched. 

#### 2.7- run all cells in the notebook

