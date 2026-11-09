import numpy as np

class Training():

    def __init__(self):
        pass


    def LD(self, output_D, M, D : int):
        sum_ = 0
        output_D_squeeze = output_D.squeeze()
        for i in range(D):
            sum_ = output_D_squeeze[i].item() * np.log2(M(i))
        return -1 * sum_

    
    def LS(W : int, H : int, cover_image, encoded_image):
        def norm(img, nb_pixels):
            sum_ = 0
            for elt in img:
                sum_ += elt * elt
            sum_ = np.sqrt(sum_)
            sum_ *= sum_
            return sum_

        size = 3 * W * H
        denom = 1 / (3 * W * H)
        dist = 0

        encoded_image_squeeze = encoded_image.squeeze()
        encoded_image_items = []
        for elt in encoded_image_squeeze:
            encoded_image_items.append(elt)

        scal = 0
        for i in range(size):
            scal += cover_image[i] * encoded_image[i]
        norme_diff = norm(cover_image, size) + 2 * scal + norm(encoded_image_items, size)

        return norme_diff


    def auto_encoding_error(output_D, M, D, W, H, cover_image, encoded_image, critic_encoded):
        return LD(output_D, M, D) + LS(W, H, cover_image, encoded_image) + critic_encoded

    
    def critic_error(critic_cover_image, critic_encoded):
        return critic_cover_image - critic_encoded