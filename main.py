from src.ai_based_method import stegano_gan
from src.evaluation import utils
from PIL import Image

tools = utils.Utils()
D = 2
a = stegano_gan.Stegano_gan(D = D, not_train = True)
# a.trainning(resume = True)

cover = Image.open("./assets/images/squirrel.jpg").convert("RGB")
hidden = Image.open("./assets/images/cat.jpg").convert("RGB")
width, height = cover.size
stego, infos = a.hide_adaptive(cover, hidden, height, width, filename = "squirrel_stego.png")
extracted = a.extract_image(stego, infos)

print("PSNR cover/stego : ", tools.psnr(cover, stego))
print("SSIM cover/stego : ", tools.ssim(cover, stego))
print("Bit accuracy hidden/extrated : ", tools.bit_accuracy(hidden, extracted))
print("RS-BPP hidden/extrated : ", tools.rs_bpp(hidden, extracted, D))