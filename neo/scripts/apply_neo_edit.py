from nerfstudio.utils.eval_utils import eval_setup
from pathlib import Path
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--load_config", type=Path,
                    help="path to config .yml file of neo model. Must have bboxes available present")

args = parser.parse_args()
config_path = args.load_config
# Load model
config, pipeline, ckpt_path, step = eval_setup(config_path, test_mode='test')

pipeline.apply_changes_hook()