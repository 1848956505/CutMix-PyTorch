import torch


class Cutout:

    def __init__(self, length=16):
        self.length = length

    def __call__(self, img):
        # img: Tensor [C, H, W]
        _, h, w = img.shape

        y = torch.randint(0, h, (1,)).item()
        x = torch.randint(0, w, (1,)).item()

        y1 = max(0, y - self.length // 2)
        y2 = min(h, y + self.length // 2)

        x1 = max(0, x - self.length // 2)
        x2 = min(w, x + self.length // 2)

        img = img.clone()
        img[:, y1:y2, x1:x2] = 0.0

        return img