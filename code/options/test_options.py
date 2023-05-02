import argparse

class TestOptions():
    def initialize(self):
        parser = argparse.ArgumentParser(description="test segmentation network")
        parser.add_argument("--reload_path", type=str, default='/LTUDA/checkpoint_toy/ours/model_best.pth')
        parser.add_argument("--result_path", type=str, default='/LTUDA/results/ours/')
        parser.add_argument("--test_path", type=str, default="/LTUDA/data/Toy dataset/test_volume.txt")
        return parser.parse_args()

