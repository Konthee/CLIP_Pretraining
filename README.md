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

Get back to the parent folder:
```python
$ cd ..
```
**2. Run the Pretraining Script**

To initiate the pretraining process, execute the Python script:
```python
$ python CLIP_Pretraining/scripts/pretraining.py
```
