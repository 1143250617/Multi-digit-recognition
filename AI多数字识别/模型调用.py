import pyautogui
import torch
from PIL import Image
from torchvision import transforms
from model import Model

def 仅支持五位整数(path_to_checkpoint_file, path_to_input_image):
    model = Model()
    model.restore(path_to_checkpoint_file)
    model.cuda()

    with torch.no_grad():
        transform = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.CenterCrop([54, 54]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        image = Image.open(path_to_input_image)
        # print(image)
        image = image.convert('RGB')
        image = transform(image)
        images = image.unsqueeze(dim=0).cuda()

        length_logits, digit1_logits, digit2_logits, digit3_logits, digit4_logits, digit5_logits = model.eval()(images)

        length_prediction = length_logits.max(1)[1]
        digit1_prediction = digit1_logits.max(1)[1]
        digit2_prediction = digit2_logits.max(1)[1]
        digit3_prediction = digit3_logits.max(1)[1]
        digit4_prediction = digit4_logits.max(1)[1]
        digit5_prediction = digit5_logits.max(1)[1]

        return length_prediction.item(),digit1_prediction.item(), digit2_prediction.item(), digit3_prediction.item(), digit4_prediction.item(), digit5_prediction.item()

if __name__ == '__main__':
    # images内有1~5张图片用于测试
   length,a,b,c,d,e = 仅支持五位整数('训练后得到的模型\\model-95000.pth', '.\\images\\1.png')
   print('length:', length)
   print('digits:', a,b,c,d,e)
   print('10代表空')

   im = pyautogui.screenshot(region=(436, 254, 459 - 436, 275 - 254))
   length, a, b, c, d, e = 仅支持五位整数('训练后得到的模型\\model-95000.pth',im)






