import torch, torchvision, pathlib
from export_wrapper import Wrapped   # your class

# set up
repo = pathlib.Path("model_repository/retail_detector/1")
repo.mkdir(parents=True, exist_ok=True)

# example input for tracing
example = torch.randn(1, 3, 320, 320)

# trace instead of script
traced = torch.jit.trace(Wrapped(), example)

# save
traced.save(repo / "model.pt")
print("✅ Torch‑traced model saved →", repo / "model.pt")
