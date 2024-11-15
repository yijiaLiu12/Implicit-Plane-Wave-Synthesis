Environment Setup
To configure the environment for this project, please refer to the augan_environment.yml for detailed instructions on setting up the necessary dependencies and environment variables.

Training Overview
This project is divided into two stages of training:

1.Parallel Training
  In this stage, the network is trained in parallel for the left and right angles. You need to execute two separate Python scripts: trainMAT_stage1.py for training the networks for the left and right angles, and trainMAT_stage2.py for training the fusion network in the second stage.
  Stage 1: trainMAT_stage1.py
    For this stage, you will train separate networks for the left and right angles. Here’s how to modify the configuration:
    
    Path for saving weights:
    Set the opt.name parameter to either 'tribranch_stage1_R' or 'tribranch_stage1_L' depending on which angle you are training for (Right or Left).
    The weights for the left and right networks will be saved separately. These will be saved under the checkpoints folder in directories named 'tribranch_stage1_R' or 'tribranch_stage1_L'.
    
    Loss file path:
    Modify the loss_file_path to specify your own path. It's recommended to set a path for saving the training loss details.
    
    Target modification:
    Inside the training loop, modify the target as follows:
    
    For the left angle: target5 = batch['left']
    For the right angle: target5 = batch['right']
  
  Stage 2: trainMAT_stage2.py
    While training the first stage network, the second stage fusion network can be trained simultaneously. This step is executed by running trainMAT_stage2.py. The trained model for this stage will be saved automatically under the tribranch_stage2 folder in the checkpoints directory.


2. Moving from Parallel to Serial Training
  Once the networks in both the left and right angles are trained and stable in Stage 1 and Stage 2, you need to prepare for serial training:
  
  Move Trained Weights:
  Manually copy the trained weights from Stage 1 into a new folder called tribranch_whole inside the checkpoints folder.
  
  Start Serial Training:
  With the weights in place, you can proceed with serial training by running the training script for the entire system.
  
  Notes on Training and Epochs：
  In our case, during the parallel stage, the training will be conducted for 50 epochs. Thus, when starting the serial training, we set it begin from epoch 50. If necessary, you can adjust the starting epoch by modifying the opt.epoch parameter in the configuration file.
