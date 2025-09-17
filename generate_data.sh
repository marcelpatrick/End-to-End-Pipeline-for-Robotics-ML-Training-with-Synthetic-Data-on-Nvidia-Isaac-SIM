#!/bin/bash

# This is the path where Isaac Sim is installed which contains the python.sh script
ISAAC_SIM_PATH="C:\isaacsim"

## Go to location of the SDG script
cd ../palletjack_sdg
SCRIPT_PATH="${PWD}/standalone_palletjack_sdg.py"
OUTPUT_WAREHOUSE="${PWD}/palletjack_data/distractors_warehouse"
OUTPUT_ADDITIONAL="${PWD}/palletjack_data/distractors_additional"
OUTPUT_NO_DISTRACTORS="${PWD}/palletjack_data/no_distractors"


## Go to Isaac Sim location for running with ./python.sh
cd $ISAAC_SIM_PATH

echo "Starting Data Generation"  

cmd //c "C:\isaacsim\python.bat" $SCRIPT_PATH --height 544 --width 960 --num_frames 2000 --distractors warehouse --data_dir $OUTPUT_WAREHOUSE

cmd //c "C:\isaacsim\python.bat" $SCRIPT_PATH --height 544 --width 960 --num_frames 2000 --distractors additional --data_dir $OUTPUT_ADDITIONAL

cmd //c "C:\isaacsim\python.bat" $SCRIPT_PATH --height 544 --width 960 --num_frames 1000 --distractors None --data_dir $OUTPUT_NO_DISTRACTORS


