import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from src.utils.grad_cam.model.excd_dd import MECACD
from torchvision import transforms
from utils import GradCAM, show_cam_on_image


def main():
    model = MECACD(bkbn_name="efficientnet_v2_m")
    weights_path = "model/model_best_mecacd.pth"
    model.load_state_dict(torch.load(weights_path, map_location="cpu")['state_dict'])
    target_layers = [model._mixing_mask[-1]]

    # model = models.vgg16(pretrained=True)
    # target_layers = [model.features]

    # model = models.resnet34(pretrained=True)
    # target_layers = [model.layer4]

    # model = models.regnet_y_800mf(pretrained=True)
    # target_layers = [model.trunk_output]

    # model = models.efficientnet_b0(pretrained=True)
    # target_layers = [model.features]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # load image
    img_path = "image/test_13_3.png"
    img_path1 = "test_13_31.png"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    img1 = Image.open(img_path1).convert('RGB')
    img1 = np.array(img1, dtype=np.uint8)
    # img = center_crop_img(img, 224)

    # [C, H, W]
    img_tensor1 = data_transform(img)
    img_tensor2 = data_transform(img1)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor1 = torch.unsqueeze(img_tensor1, dim=0)
    input_tensor2 = torch.unsqueeze(img_tensor2, dim=0)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    target_category = 0  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog

    grayscale_cam = cam(input_tensor1=input_tensor1,input_tensor2=input_tensor2, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    plt.show()


if __name__ == '__main__':
    main()
