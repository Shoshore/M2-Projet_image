from src.ai_based_method import stegano_gan
from src.evaluation import psnr
from PIL import Image
a = stegano_gan.Stegano_gan(2)
# a.trainning(resume = False)
a.load_model()
cover = Image.open("./assets/images/squirrel.jpg").convert("RGB")
hidden = Image.open("./assets/images/cat.jpg").convert("RGB")
height, width = cover.size
stego, infos = a.hide_adaptive("./assets/images/squirrel.jpg", "./assets/images/cat.jpg", height, width, filename = "squirrel_stego.png")
extracted = a.extract_image(stego, infos)

print(psnr.psnr(cover, stego))
print(psnr.psnr(hidden, extracted))
