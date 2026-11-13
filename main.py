from src.ai_based_method import stegano_gan
from src.evaluation import utils
from PIL import Image

tools = utils.Utils()

a = stegano_gan.Stegano_gan(D = 2, not_train = True)
# a.trainning(resume = False)

cover = Image.open("./assets/images/squirrel.jpg").convert("RGB")
hidden = Image.open("./assets/images/cat.jpg").convert("RGB")
height, width = cover.size
stego, infos = a.hide_adaptive(cover, hidden, height, width, filename = "squirrel_stego.png")
extracted = a.extract_image(stego, infos)

print("PSNR cover : ", tools.psnr(cover, stego))
print("PSNR hidden : ", tools.psnr(hidden, extracted))
print("SSIM cover : ", tools.ssim(cover, stego))
print("SSIM hidden : ", tools.ssim(hidden, extracted))
