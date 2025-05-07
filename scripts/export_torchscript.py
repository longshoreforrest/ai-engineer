import torch, torchvision, pathlib

repo = pathlib.Path("model_repository/retail_detector/1")
repo.mkdir(parents=True, exist_ok=True)

weights = torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
model   = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=weights)
model.eval()

ts = torch.jit.script(model)          # turn it into TorchScript
ts.save(repo / "model.pt")            # overwrite if one existed
print("TorchScript model saved:", repo / "model.pt")
