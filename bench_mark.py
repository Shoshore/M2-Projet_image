from src.ai_based_method import stegano_gan
from src.evaluation import utils
from PIL import Image
from os import listdir
from os.path import splitext, join
tools = utils.Utils()
D = 2
FILE = "./assets/images/"
a = stegano_gan.Stegano_gan(D = D, not_train = True)
# a.trainning(resume = True)

mean_psnr = 0
mean_ssim = 0
mean_bit_acc = 0
mean_rs_bpp = 0
nb_img = 0
hidden = Image.open("./assets/images/cat.jpg").convert("RGB")
wh, hh = hidden.size
for name in sorted(listdir(FILE)):
    path = join(name)
    out_name, _ = splitext(name)
    if path == "cat.jpg":
        continue
    nb_img += 1
    cover = Image.open(FILE + path).convert("RGB")
    w, h = cover.size
    out_name += '_stego.png'
    stego, infos = a.hide_adaptive(cover, hidden, h, w, filename=out_name)
    extracted = a.extract_image(stego, infos)
    we, he = extracted.size  
    hidden_comp = hidden.resize((we, he))
    mean_psnr += tools.psnr(cover, stego)
    mean_ssim += tools.ssim(cover, stego)
    mean_bit_acc += tools.bit_accuracy(hidden_comp, extracted)
    mean_rs_bpp += tools.rs_bpp(hidden_comp, extracted, D)

mean_psnr /= nb_img
mean_ssim /= nb_img
mean_bit_acc /= nb_img
mean_rs_bpp /= nb_img
print("Nb image : ", nb_img)
print("Bit accuracy hidden/extrated : ", mean_bit_acc)
print("RS-BPP hidden/extrated : ", mean_rs_bpp)
print("PSNR cover/stego : ", mean_psnr)
print("SSIM cover/stego : ", mean_ssim)



# cover = Image.open("./assets/images/squirrel.jpg").convert("RGB")

# width, height = cover.size
# stego, infos = a.hide_adaptive(cover, hidden, height, width)
# extracted = a.extract_image(stego, infos)

# print("PSNR cover/stego : ", tools.psnr(cover, stego))
# print("SSIM cover/stego : ", tools.ssim(cover, stego))
# print("Bit accuracy hidden/extrated : ", tools.bit_accuracy(hidden, extracted))
# print("RS-BPP hidden/extrated : ", tools.rs_bpp(hidden, extracted, D))