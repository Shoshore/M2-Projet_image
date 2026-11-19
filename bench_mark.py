from src.ai_based_method import stegano_gan
from src.evaluation import utils
from PIL import Image
from os import listdir
from os.path import splitext, join
from math import sqrt
tools = utils.Utils()

D = 2
FILE = "./assets/images/"
NB_IMAGE = 5
SECRET_LIST = sorted(listdir(FILE + "/secret"))
COVER_LIST = sorted(listdir(FILE + "/cover"))
MODE = [18, 9, 6, 5, 4, 3]

actual = 0
gan = stegano_gan.Stegano_gan(D = D, not_train = True, max_epoch = 32)
# a.trainning(resume = True)


stats_stegano = {
    "mean_psnr_stego" : 0,
    "mean_ssim_stego" : 0,
    "mean_psnr_extract" : 0,
    "mean_ssim_extract" : 0,
    "mean_bit_acc_extract" : 0,
    "mean_rs_bpp_extract" : 0,
}

def redim(secret_img, cover_w, cover_h):
    ratio = MODE[D - 1]
    sw, sh = secret_img.size
    target_pixels = (cover_w * cover_h) / ratio
    scale = sqrt(target_pixels / (sw * sh))
    new_w = max(1, round(sw * scale))
    new_h = max(1, round(sh * scale))
    return secret_img.resize((new_w, new_h), Image.LANCZOS)


for i in range(NB_IMAGE):
    cover_name = COVER_LIST[i]
    cover_path = join(cover_name)
    cover_img = Image.open(FILE + "/cover/" + cover_path).convert("RGB")
    cover_w, cover_h = cover_img.size

    stego_out_name, _ = splitext(cover_name)
    stego_out_name += '_stego.png'

    secret_name = SECRET_LIST[i]
    secret_path = join(secret_name)
    secret_img = Image.open(FILE + "/secret/" + secret_path).convert("RGB")

    secret_redim_img = redim(secret_img, cover_w, cover_h)

    stego_img, infos = gan.hide_adaptive(cover_img, secret_redim_img, cover_h, cover_w, filename = stego_out_name)
    if stego_img == None:
        continue
    actual += 1

    extracted_img = gan.extract_image(stego_img, infos)

    stats_stegano["mean_psnr_stego"] += tools.psnr(cover_img, stego_img)
    stats_stegano["mean_ssim_stego"] += tools.ssim(cover_img, stego_img)

    stats_stegano["mean_psnr_extract"] += tools.psnr(secret_redim_img, extracted_img)
    stats_stegano["mean_ssim_extract"] += tools.ssim(secret_redim_img, extracted_img)
    stats_stegano["mean_bit_acc_extract"] += tools.bit_accuracy(secret_redim_img, extracted_img)
    stats_stegano["mean_rs_bpp_extract"] += tools.rs_bpp(secret_redim_img, extracted_img, D)



print("Nb image : ", actual)
for cle in stats_stegano:
    stats_stegano[cle]  /=  actual
    print(cle," : ", stats_stegano[cle])