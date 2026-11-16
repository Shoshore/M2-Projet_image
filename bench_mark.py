from src.ai_based_method import stegano_gan
from src.evaluation import utils
from PIL import Image
from os import listdir
from os.path import splitext, join
tools = utils.Utils()
D = 2
FILE = "./assets/images/"

a = stegano_gan.Stegano_gan(D = D, not_train = True, max_epoch = 75)
# a.trainning(resume = True)

mean_psnr_stego = 0
mean_psnr_extract =0
mean_ssim_stego = 0
mean_ssim_extract = 0
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
    mean_psnr_stego += tools.psnr(cover, stego)
    mean_psnr_extract += tools.psnr(hidden_comp, extracted)
    mean_ssim_stego += tools.ssim(cover, stego)
    mean_ssim_extract += tools.ssim(hidden_comp, extracted)
    mean_bit_acc += tools.bit_accuracy(hidden_comp, extracted)
    mean_rs_bpp += tools.rs_bpp(hidden_comp, extracted, D)

mean_psnr_stego /= nb_img
mean_psnr_extract /= nb_img
mean_ssim_extract /= nb_img
mean_ssim_stego /= nb_img
mean_bit_acc /= nb_img
mean_rs_bpp /= nb_img
print("Nb image : ", nb_img)

print("Bit accuracy hidden/extrated : ", mean_bit_acc)
print("RS-BPP hidden/extrated : ", mean_rs_bpp)

print("PSNR extract/hidden : ", mean_psnr_extract)
print("SSIM extract/hidden : ", mean_ssim_extract)

print("PSNR cover/stego : ", mean_psnr_stego)
print("SSIM cover/stego : ", mean_ssim_stego)