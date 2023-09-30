# Pretraining CLIP Model 
model : [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)


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
