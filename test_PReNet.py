import argparse
import cv2
import time

from networks import *
from utils import *

parser = argparse.ArgumentParser(description="PReNet_Test")
parser.add_argument("--logdir", type=str, default="logs/PReNet6/", help='path to model and log files')
parser.add_argument("--data_path", type=str, default="/media/r/BC580A85580A3F20/dataset/rain/peku/Rain100H/rainy",
                    help='path to training data')
parser.add_argument("--save_path", type=str, default="/home/r/works/derain_arxiv/release/results/PReNet",
                    help='path to save results')
parser.add_argument("--use_GPU", type=bool, default=False, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages')
opt = parser.parse_args()

if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id


def main():
    print(opt.use_GPU)
    os.makedirs(opt.save_path, exist_ok=True)

    # Build model
    print('Loading model ...\n')
    model = PReNet(opt.recurrent_iter, opt.use_GPU)
    print_network(model)
    if opt.use_GPU:
        model = model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net_latest.pth'), map_location=device))
    model.eval()

    time_test = 0
    count = 0
    for img_name in os.listdir(opt.data_path):
        if is_image(img_name):
            img_path = os.path.join(opt.data_path, img_name)

            # input image
            y = cv2.imread(img_path)
            b, g, r = cv2.split(y)
            y = cv2.merge([r, g, b])
            # y = cv2.resize(y, (int(500), int(500)), interpolation=cv2.INTER_CUBIC)

            y = normalize(np.float32(y))
            y = np.expand_dims(y.transpose(2, 0, 1), 0)     # np.expand_dims(x, 0): 在 x 前面添加一个新的轴
            y = Variable(torch.Tensor(y))

            if opt.use_GPU:
                y = y.cuda()

            with torch.no_grad():
                if opt.use_GPU:
                    # 等待所有 CUDA 操作全部完成后，再进行推理。这是为了更精确地计数。
                    torch.cuda.synchronize()
                start_time = time.time()

                out, _ = model(y)
                '''
                clamp函数将输入out张量中的每个元素限制在区间[0., 1.]内。
                也就是说，如果out中的某个元素小于0，那么它会被设置为0；
                如果out中的某个元素大于1，那么它会被设置为1；
                如果out中的某个元素在0和1之间，那么它保持不变。
                '''
                out = torch.clamp(out, 0., 1.)

                if opt.use_GPU:
                    torch.cuda.synchronize()
                end_time = time.time()
                dur_time = end_time - start_time
                time_test += dur_time

                print(img_name, ': ', dur_time)

            if opt.use_GPU:
                save_out = np.uint8(255 * out.data.cpu().numpy().squeeze())  # back to cpu
            else:
                save_out = np.uint8(255 * out.data.numpy().squeeze())

            save_out = save_out.transpose(1, 2, 0)
            b, g, r = cv2.split(save_out)
            save_out = cv2.merge([r, g, b])

            cv2.imwrite(os.path.join(opt.save_path, img_name), save_out)

            count += 1

    print('Avg. time:', time_test / count)


if __name__ == "__main__":
    main()
