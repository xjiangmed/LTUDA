import argparse

class TestOptions():
    def initialize(self):
        parser = argparse.ArgumentParser(description="test segmentation network")
        parser.add_argument("--reload_path", type=str, default='./checkpoint/CDA_PDA/model_best.pth')
        parser.add_argument("--result_path", type=str, default='./results/ours/')
        parser.add_argument("--test_path", type=str, default="./data/Toy dataset/test_volume.txt")
        return parser.parse_args()

