import argparse

class TestOptions():
    def initialize(self):
        parser = argparse.ArgumentParser(description="test segmentation network")
        parser.add_argument('--model', type=str, default='unet', help='model_name')
        parser.add_argument("--reload_path", type=str, default='../checkpoint/CDA_PDA/ema_model_best.pth')
        parser.add_argument("--result_path", type=str, default='../results/CDA_PDA/')
        parser.add_argument("--test_path", type=str, default="../data/Toy dataset/test_volume.txt")
        parser.add_argument("--linear_classifier", type=bool, default=False)
        parser.add_argument("--lp_classifier", type=bool, default=False)
        parser.add_argument("--ulp_classifier", type=bool, default=True)
        parser.add_argument("--post", type=bool, default=True)
        return parser.parse_args()