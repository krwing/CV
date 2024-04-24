from min_dalle import MinDalle
import torch
from PIL import Image

model = MinDalle( #Initialize the model
    models_root='./pretrained',
    dtype=torch.float16,
    device='cuda',
    is_mega=True,
    is_reusable=True
)
print("="*100)
image = model.generate_image(
    text='A huge deserted building complex',#text prompt
    seed=-1,
    grid_size=1,
    is_seamless=False,
    temperature=1,
    top_k=4,
    supercondition_factor=32,
    is_verbose=False
)
print("="*100)


newsize = (512, 512)
image = image.resize(newsize)
image.save('dall.jpg')

