from matplotlib import pyplot as plt
import torch
import torchvision.transforms.functional as F

from metrics import dice, hd95


def show_result(input, mask, y_prob, include_dice=False, include_hd95=False):
    plt.figure(figsize=(15, 5))

    plt.subplot(141)
    plt.imshow(F.to_pil_image(input))
    plt.title('MRI')
    plt.axis('off')

    plt.subplot(142)
    plt.imshow(F.to_pil_image(mask), cmap='gray')
    plt.title('Segmentation Mask')
    plt.axis('off')

    plt.subplot(143)
    plt.imshow(y_prob.cpu().detach().numpy(), cmap='gray', vmin=0, vmax=1)
    plt.title('Output Probabilities')
    plt.axis('off')
    # plt.colorbar(y_prob, ax=plt.gca(), fraction=0.046, pad=0.04)

    plt.subplot(144)
    # y = torch.round(y_prob)
    y = (y_prob > 0.2).float()
    plt.imshow(F.to_pil_image(y), cmap='gray')
    title = 'Output'
    if include_dice:
        title += f'\nDice = {dice(y, mask):.3f}'
    if include_hd95:
        title += f'\nHD95 = {hd95(y, mask):.3f}'
    plt.title(title)
    plt.axis('off')
    plt.axis('off')

    plt.show()


