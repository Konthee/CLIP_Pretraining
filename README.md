# Pretraining CLIP Model 
The objective of this repository is pretr

model : [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)

CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing for the task, similarly to the zero-shot capabilities of GPT-2 and 3. We found CLIP matches the performance of the original ResNet50 on ImageNet “zero-shot” without using any of the original 1.28M labeled examples, overcoming several major challenges in computer vision.




![CLIP.png](https://github.com/openai/CLIP/blob/main/CLIP.png)

## Setup
Clone Repository
```python
$ git clone https://github.com/Konthee/CLIP_Pretraining.git
```
Change directory to repository folder

```python
$ cd CLIP_Pretraining
```
Install module on your environment from [pyproject.toml](https://github.com/Konthee/CLIP_Pretraining/blob/main/pyproject.toml)
```python
$ pip install -e .
```
## How to RUN 
Back from repository folder 
```python
$ cd ..
```
Running a Python script with an existing environment
```python
$ python CLIP_Pretraining/scripts/pretraining.py
```
