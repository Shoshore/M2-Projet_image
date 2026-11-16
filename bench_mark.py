from src.ai_based_method import stegano_gan
from src.evaluation import utils
from PIL import Image
from os import listdir
from os.path import splitext, join
tools = utils.Utils()
D = 2
FILE = "./assets/images/cover"

a = stegano_gan.Stegano_gan(D = D, not_train = True, max_epoch = 75)
# a.trainning(resume = True)

nb_img = 0
stats_stegano = {
    "mean_psnr_stego" : 0,
    "mean_ssim_stego" : 0,
    "mean_psnr_extract" : 0,
    "mean_ssim_extract" : 0,
    "mean_bit_acc_extract" : 0,
    "mean_rs_bpp_extract" : 0,
}

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

    stats_stegano["mean_psnr_stego"] += tools.psnr(cover, stego)
    stats_stegano["mean_ssim_stego"] += tools.ssim(cover, stego)

    stats_stegano["mean_psnr_extract"] += tools.psnr(hidden_comp, extracted)
    stats_stegano["mean_ssim_extract"] += tools.ssim(hidden_comp, extracted)
    stats_stegano["mean_bit_acc_extract"] += tools.bit_accuracy(hidden_comp, extracted)
    stats_stegano["mean_rs_bpp_extract"] += tools.rs_bpp(hidden_comp, extracted, D)

print("Nb image : ", nb_img)
for cle in stats_stegano:
    stats_stegano[cle]  /=  nb_img
    print(cle," : ", stats_stegano[cle])