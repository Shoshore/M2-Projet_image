from numpy import mean, log10, array, float64, ndarray, unpackbits, uint8, concatenate, zeros, asarray, atleast_1d, hstack
from PIL.Image import Image, open
from skimage.metrics import structural_similarity
from sealwatch import chi2, spa, ws, spam, crm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
class Utils:

    def __init__(self):
        pass


    def _to_array(self, img) -> ndarray:
        if isinstance(img, ndarray):
            arr = img
        elif isinstance(img, Image):
            arr = array(img.convert("RGB"))
        else:
            arr = array(open(img).convert("RGB"))
        return arr.astype(float64)


    def psnr(self, cover_image, stego_image, max_value: float = 255.0) -> float:
        a = self._to_array(cover_image)
        b = self._to_array(stego_image)

        if a.shape != b.shape:
            raise ValueError(f"Tailles différentes : {a.shape} vs {b.shape}. ""Redimensionne la cover à la taille de la stego avant de comparer.")

        mse = mean((a - b) ** 2)
        if mse == 0:
            return float("inf")
        return float(10 * log10(max_value ** 2 / mse))

    
    def ssim(self, cover_image, stego_image, max_value: float = 255.0) -> float:
        a = self._to_array(cover_image)
        b = self._to_array(stego_image)

        if a.shape != b.shape:
            raise ValueError(f"Tailles différentes : {a.shape} vs {b.shape}. ""Redimensionne la cover à la taille de la stego avant de comparer.")

        return float(structural_similarity(
            a, b,
            data_range=max_value,
            channel_axis=2,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
        ))


    def bit_accuracy(self, hidden_image, extracted_image):
        a = self._to_array(hidden_image).astype(uint8)
        b = self._to_array(extracted_image).astype(uint8)
        if a.shape != b.shape:
            raise ValueError(f"Tailles différentes : {a.shape} vs {b.shape}. " "Redimensionne l'image cachée à la taille de l'extraite avant de comparer.")
        bits_a = unpackbits(a)
        bits_b = unpackbits(b)
        return float(mean(bits_a == bits_b))


    def rs_bpp(self, hidden_image, extracted_image, D: float = 2.0):
        a = self._to_array(hidden_image).astype(uint8)
        b = self._to_array(extracted_image).astype(uint8)
        if a.shape != b.shape:
            raise ValueError(f"Tailles différentes : {a.shape} vs {b.shape}. " "Redimensionne l'image cachée à la taille de l'extraite avant de comparer.")
        acc_pct = self.bit_accuracy(hidden_image, extracted_image) * 100.0
        p = 1.0 - (acc_pct / 100.0)
        return float(max(0.0, (1.0 - 2.0 * p) * D))


    def score_steg(self, img):
        arr = array(img.convert("RGB"), dtype=uint8)
        chans = [arr[:, :, c] for c in range(3)]

        return {
            "chi2": float(mean([chi2.attack(c)[1] for c in chans])),
            "spa":  float(mean([spa.attack(c) for c in chans])),
            "ws":   float(mean([ws.attack(c) for c in chans])),
        }

    def _to_vec(self, res):
        parts = []

        def walk(v):
            if isinstance(v, dict):
                for x in v.values():
                    walk(x)
            elif isinstance(v, (tuple, list)):
                for x in v:
                    walk(x)
            else:
                parts.append(atleast_1d(asarray(v, dtype=float64)).ravel())

        walk(res)
        return hstack(parts)


    def features_spam(self, img):
        arr = array(img.convert("RGB"), dtype=uint8)
        vecs = [self._to_vec(spam.extract(arr[:, :, c])) for c in range(3)]
        return concatenate(vecs) 


    def features_crm(self, img):
        arr = array(img.convert("RGB"), dtype=uint8)
        return self._to_vec(crm.extract(arr))


    def _auroc_cv(self, feats_cover: list, feats_stego: list, n_splits: int = 5):
        n = len(feats_cover)
        X = array(feats_cover + feats_stego, dtype=float64)
        y = array([0] * n + [1] * n)
        groups = array(list(range(n)) * 2)

        n_splits = min(n_splits, n)
        scores = zeros(len(y))
        fold_auc = []
        for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000))
            clf.fit(X[tr], y[tr])
            scores[te] = clf.decision_function(X[te])
            fold_auc.append(roc_auc_score(y[te], scores[te]))

        return (float(roc_auc_score(y, scores)), float(mean(fold_auc)), float(array(fold_auc).std()))


    def auroc(self, scores_cover: list, scores_stego: list, spam_cover: list = None, spam_stego: list = None, crm_cover: list = None, crm_stego: list = None) -> dict:
        res = {}

        labels = [0] * len(scores_cover) + [1] * len(scores_stego)
        for name in scores_cover[0]:
            scores = [s[name] for s in scores_cover] + [s[name] for s in scores_stego]
            res[name] = {"auroc": float(roc_auc_score(labels, scores))}

        if spam_cover and spam_stego:
            a, m, s = self._auroc_cv(spam_cover, spam_stego)
            res["spam"] = {"auroc": a, "fold_mean": m, "fold_std": s}
        if crm_cover and crm_stego:
            a, m, s = self._auroc_cv(crm_cover, crm_stego)
            res["crm"] = {"auroc": a, "fold_mean": m, "fold_std": s}

        print(f"{'test':<8} {'AUROC':>8}   plis (moyenne ± écart-type)")
        for name, r in res.items():
            extra = f"{r['fold_mean']:.4f} ± {r['fold_std']:.4f}" if "fold_mean" in r else "-"
            print(f"{name:<8} {r['auroc']:>8.4f}   {extra}")

        return res