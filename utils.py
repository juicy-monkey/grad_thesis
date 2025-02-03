from matplotlib import pyplot as plt
import torch
import torchvision.transforms.functional as F


def show_result(input, mask, y_prob):
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
    y = torch.round(y_prob)
    plt.imshow(F.to_pil_image(y), cmap='gray')
    title = 'Output'
    plt.title(title)
    plt.axis('off')
    plt.axis('off')

    plt.show()


