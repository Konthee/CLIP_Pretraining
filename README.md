# Pretraining CLIP Model 
The purpose of this repository is to facilitate the creation and management of a streamlined pipeline for pretrainning CLIP model. The data available in the [`lambdalabs/pokemon-blip-captions`](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions) dataset

model : [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32)

CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing for the task, similarly to the zero-shot capabilities of GPT-2 and 3. We found CLIP matches the performance of the original ResNet50 on ImageNet “zero-shot” without using any of the original 1.28M labeled examples, overcoming several major challenges in computer vision.




![CLIP.png](https://github.com/openai/CLIP/blob/main/CLIP.png)

## Setup Instructions
**1. Clone the Repository**

Execute the following command to clone the repository:
```python
$ git clone https://github.com/Konthee/CLIP_Pretraining.git
```

**2. Navigate to the Repository Folder**

Change your directory to the cloned repository's folder:
```python
$ cd CLIP_Pretraining
```

**3. Install the Required Modules**

Use the following command to install necessary modules from the [pyproject.toml](https://github.com/Konthee/CLIP_Pretraining/blob/main/pyproject.toml):
```python
$ pip install -e .
```

## Execution Steps

**1. Navigate to the Parent Directory**

Firstly, move back to the parent directory:
```python
$ cd ..
```
**2. Run the Pretraining Script**

Ensure you're using the appropriate environment, then execute the pretraining script::
```python
$ python CLIP_Pretraining/scripts/pretraining.py
```

There are 2 step in [`pretraining.py`](https://github.com/Konthee/CLIP_Pretraining/blob/main/scripts/pretraining.py)
- Firstly, execute the [`dataset_preparation.py`](https://github.com/Konthee/CLIP_Pretraining/blob/main/scripts/dataset_preparation.py) script to download the dataset. This script will replace  `image` and `text` column to be `input_ids`, `attention_mask`,and `pixel_values` column and also split the data into training and validation sets at an 85% and 15% respectively. you'll notice an output folder named Dataset in the parent directory
- Secondly, the process involves pretraining your model based on the configuration settings specified in [`pretraining_configuration`](https://github.com/Konthee/CLIP_Pretraining/tree/main/scripts/pretraining_configuration). Upon completion, an output folder will be generated, encompassing elements such as checkpoints, outputs, lightning_logs, and wandb.


#### Here's a breakdown of Second Step in the `pretraining.py` script:

1. **Model Configuration**:
   The script imports necessary modules and then defines the training procedure under the `clip_pretraining` function. This function leverages the given configuration settings including:
   - Model configuration
   - Trainer configuration
   - Dataset parameters
   - Optimizer and scheduler details
   - Checkpointing and logging preferences

2. **Data Loading**:
   The training and validation datasets are loaded using `load_from_disk`, ensuring they are in the appropriate torch format. Dataloaders are then created for both datasets with specified batch sizes and workers.

3. **Model Initialization**:
   The `CLIPModelPL` class is an instance of the PyTorch Lightning module tailored for the CLIP model. Within this class:
   - The model architecture is defined, including conditions to freeze specific layers or to apply a better transformer if specified.
   - Optimizer and scheduler configurations are set.
   - Training and validation steps are described, with loss being the primary metric.

4. **Callbacks & Loggers**:
   Callback functions like model checkpointing and early stopping are set based on the configuration. Additionally, logging configurations for tools like WandB and TensorBoard are initialized.

5. **Training**:
   A PyTorch Lightning `Trainer` is initialized with all the specified configurations. The training process is then initiated with `trainer.fit`, passing in the model and the data loaders.

## Setting Up Configuration Files with Hydra
[Hydra](https://hydra.cc/docs/1.3/intro/) is a powerful configuration management tool developed by Facebook Research. It allows users to compose and manage complex configuration setups with ease, making it simpler to develop and experiment with different configurations.

### **Basic Usage**

Hydra provides an elegant solution for managing configurations, especially for machine learning tasks where various components may have different settings. By using Hydra, one can seamlessly change configurations, such as model parameters, optimization techniques, or logging settings, all through configuration files. Here's how to set it up for a pretraining task:

### Directory and Main Configuration File

1. **Directory Structure**: Create a directory specifically for your pretraining configurations. Name it `pretraining_configuration`.

2. **Main Configuration File**: Inside this directory, establish a primary YAML configuration file named `default_config.yaml`.

`pretraining_configuration/default_config.yaml`:
```yaml
defaults:
  - model: default_model
  - optimizer: adam_optimizer    # adam_optimizer or adamw_optimizer 
  - scheduler: cosine_lr          # 'none' , cosine_lr , steplr
  - trainer: default_trainer
  - dataset: default_dataset
  - checkpointing: default_checkpointing
  - logging: default_logging

```

The `defaults` list indicates which configurations to use by default. For instance, when the pretraining task begins, unless specified otherwise, it'll utilize settings from `default_trainer.yaml` for the model, `adam_optimizer.yaml` for the optimizer, and so forth.

### Component Configuration Files

For each component listed under `defaults`, relate a corresponding YAML file:

- #### `pretraining_configuration/trainer/default_trainer.yaml`:

```yaml
batch_size: 16
max_steps: 1e6
max_epochs: 100 
dropout: 0.2
precision: 32-true             # 16-mixed 32-true  
accelerator: "auto"              # Options: "cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto"
accumulate_grad_batches: 1       # 16*2 =32
better_transformer : False        # Use bettertransformer if True
freeze_image_encoder: True       # Freeze image encoder layer weights if True

resume_from_checkpoint:
 use: False
 checkpoint_path: "checkpoints/checkpoint_test.ckpt"

early_stopping:
 use: True
 patience: 5
 min_delta: 0.01
 verbose: True
 mode: "min" 
```

#### Explanation:

1. **batch_size**: Specifies the number of samples used in one iteration. A smaller batch size usually offers more frequent model updates and might converge faster, albeit noisily, while a larger one provides more stable and less frequent updates.

2. **max_steps**: Maximum number of steps (batches) the trainer will run. If set, `max_epochs` will be ignored.

3. **max_epochs**: Specifies how many times the entire dataset will be passed through the model. One epoch means one forward and backward pass of all training samples.

4. **dropout**: Represents the probability of dropping out a given neuron during training, helping in the regularization of the model.

5. **precision**: Determines the floating-point precision. `16-mixed` uses mixed precision with 16 and 32 bits, which can speed up training on supported GPUs. `32-true` uses pure 32-bit precision.

6. **accelerator**: Specifies the hardware accelerator type. Can be `cpu`, `gpu`, `tpu`,`auto` or etc lets the trainer choose the best available option.

7. **accumulate_grad_batches**: Determines how many batches to process before performing a backward/update pass. This can effectively mimic a larger batch size.

8. **better_transformer**: If set to `True`, the model uses an improved transformer variant.

9. **freeze_image_encoder**: If `True`, the weights of the image encoder layer remain constant during training, i.e., they won't be updated.

10. **resume_from_checkpoint**:
    - **use**: training resume from a checkpoint if set to `True`.
    - **checkpoint_path**: If resuming, this path points to the specific checkpoint file.

11. **early_stopping**:
    - **use**: Activates early stopping if set to `True`.
    - **patience**: Number of epochs with no improvement after which training will be stopped.
    - **min_delta**: Minimum change in the monitored quantity to qualify as an improvement.
    - **verbose**: If `True`, it'll print log messages for early stopping.
    - **mode**: If “min”, training will stop when the quantity monitored has stopped decreasing; if “max”, it'll stop when the quantity monitored has stopped increasing.
  
- #### `pretraining_configuration/optimizer/adam_optimizer.yaml`: 
```yaml
type: "Adam"       
lr: 0.0004          
weight_decay: 0.2   
betas: [0.9, 0.98]  
eps: 0.000001        
```
Here, you can specify settings related to the Adam optimizer like learning rate, weight decay, beta values, and epsilon. 


- #### ... Similarly, configurations for model, trainer, dataset, checkpointing, and logging.


### Overriding Configurations

A significant advantage of Hydra is the ability to override specific configurations directly via the command line:

```python
python CLIP_Pretraining/scripts/pretraining.py scheduler=steplr optimizer=adam_optimizer
```

This command would use configurations from `steplr.yaml` and `adam_optimizer.yaml` instead of the defaults.


