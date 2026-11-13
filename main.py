from src.ai_based_method import stegano_gan
from src.evaluation import psnr
from PIL import Image
a = stegano_gan.Stegano_gan(2)
# a.trainning(resume = False)
a.load_model()

stego, infos = a.hide_adaptive("./assets/images/squirrel.jpg", "./assets/images/cat.jpg", 672, filename = "squirrel_stego.png")
cover = Image.open("./assets/images/squirrel.jpg").convert("RGB").resize((672, 672))
print(psnr.psnr(cover, stego))
