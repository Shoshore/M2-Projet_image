from src.ai_based_method import stegano_gan
from src.evaluation import utils
from PIL import Image
from os import listdir
from os.path import splitext, join
tools = utils.Utils()
D = 2
FILE = "./assets/images/"

a = stegano_gan.Stegano_gan(D = D, not_train = True, max_epoch = 32)
# a.trainning(resume = True)

nb_img = 0
stats_stegano = {
    "mean_psnr_stego" : 0,
    "mean_ssim_stego" : 0,
    "mean_psnr_extract" : 0,
    "mean_ssim_extract" : 0,
    "mean_bit_acc_extract" : 0,
    "mean_rs_bpp_extract" : 0,
    "size_min_ratio" : 100,
    "size_min_refuse_ratio" : 0,
}


for hid_name in sorted(listdir(FILE)):
    path_2 = join(hid_name)
    hidden = Image.open(FILE + path_2).convert("RGB")
    wh, hh = hidden.size
    k = wh * hh

    for name in sorted(listdir(FILE)):
        path = join(name)
        out_name, _ = splitext(name)
        if path == path_2:
            continue
        
        cover = Image.open(FILE + path).convert("RGB")
        w, h = cover.size
        out_name += '_stego.png'
        ratio = w * h / k
        stego, infos = a.hide_adaptive(cover, hidden, h, w, filename=out_name)
        if stego == None:
            if stats_stegano["size_min_refuse_ratio"] < ratio:
                stats_stegano["size_min_refuse_ratio"] = ratio
            continue
        nb_img += 1

        extracted = a.extract_image(stego, infos)

        stats_stegano["mean_psnr_stego"] += tools.psnr(cover, stego)
        stats_stegano["mean_ssim_stego"] += tools.ssim(cover, stego)

        stats_stegano["mean_psnr_extract"] += tools.psnr(hidden, extracted)
        stats_stegano["mean_ssim_extract"] += tools.ssim(hidden, extracted)
        stats_stegano["mean_bit_acc_extract"] += tools.bit_accuracy(hidden, extracted)
        stats_stegano["mean_rs_bpp_extract"] += tools.rs_bpp(hidden, extracted, D)
        
        
        if stats_stegano["size_min_ratio"] > ratio:
            stats_stegano["size_min_ratio"] = ratio



print("Nb image : ", nb_img)
for cle in stats_stegano:
    if cle not in ["size_min_ratio", "size_min_refuse_ratio"]:
        stats_stegano[cle]  /=  nb_img
    print(cle," : ", stats_stegano[cle])