import torch
import torch.nn as nn
import torch.optim as optim
import torch_directml
from torchvision import datasets, transforms, models

# Configurar o dispositivo DirectML
device = torch_directml.device()

from ultralytics import YOLO


model = YOLO("yolov10n.yaml")

model.train(data="./data.yaml", epochs=100, imgsz=640, model=model)

metrics = model.val()
path = model.export(format="onnx")  