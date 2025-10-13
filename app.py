from fastapi import FastAPI, File, UploadFile
from torchvision import models, transforms
from PIL import Image
import torch

app = FastAPI()

# Load pretrained ResNet50
model = models.resnet50(pretrained=True)
model.eval()

# Load ImageNet class labels (1,000 labels)
with open("imagenet_classes.txt") as f:
    imagenet_classes = [line.strip() for line in f.readlines()]

# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Open image
    image = Image.open(file.file).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0)

    # Forward pass
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # Get top-5 predictions
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    results = []
    for i in range(top5_prob.size(0)):
        results.append({
            "class_id": int(top5_catid[i].item()),
            "class_name": imagenet_classes[top5_catid[i]],
            "probability": float(top5_prob[i].item())
        })

    return {"predictions": results}
