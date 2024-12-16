import json
import yaml

from pathlib import Path
from argparse import ArgumentParser

from nerfstudio.engine.trainer import TrainerConfig

from neo.utils.pytex import PyTex


if __name__ == "__main__":

    # Argument parser
    parser = ArgumentParser()
    parser.add_argument("--load_config", type=Path,
                        help="path to config .yml file of neo model. Must have bboxes available present")

    args = parser.parse_args()

    # Load config
    config_path = args.load_config
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    assert isinstance(config, TrainerConfig)

    # Find results paths
    results_path = Path(str(config.get_base_dir().parents[1]).replace("outputs", "eval"))

    results_paths = []
    for path in results_path.glob("*"):
        results_paths.append((path.name, Path(f"{path}/{config.timestamp}/results.json")))

    res_keys = ["full", "cropped"]
    results = {}
    for res_key in res_keys:
        results[res_key] = {}
    for method_name, path in results_paths:
        if path.exists():
            for res_key in res_keys:
                result = json.loads(path.read_text())[res_key]
                result["Method name"] = method_name
                for key in result.keys():
                    if key not in results[res_key].keys():
                        results[res_key][key] = [result[key]]
                    else:
                        results[res_key][key].append(result[key])

    # # Generating tables
    tex = PyTex()

    columns = ["Method name", "PSNR[dB]+", "PSNR$_M$[dB]+", "MAE[m]-", "MAE$_M$[m]-"]
    name_map = {"replace-nerf": "Our", "nerfacto-nerfiller": "Na\\\"{i}ve", "grid-prior-du": "Nerfiller" }
    digits = ([2, 2, 3, 2], [2, 2, 6, 6])

    fig_name = f"{config_path.parents[4].name}"
    fig_name = (fig_name[0].upper() + fig_name[1:]).replace("_", " ")

    for res_key, digit in zip(res_keys, digits):
        tex.table(columns, results[res_key], 'lllll', digit, caption_title=fig_name, caption_text=res_key[0].upper() + res_key[1:], name_map=name_map)