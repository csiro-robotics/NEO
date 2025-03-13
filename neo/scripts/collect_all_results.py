import json

from pathlib import Path
from argparse import ArgumentParser

from neo.utils.pytex import PyTex


if __name__ == "__main__":

    # Argument parser
    parser = ArgumentParser()
    parser.add_argument("--path", type=Path,
                        help="path to folder containint results files.")

    args = parser.parse_args()

    # Find results paths
    scenes = {}
    for path in args.path.rglob("*.json"):
        relative_path = path.relative_to(args.path)
        root = relative_path.parts[0]
        if root not in scenes:
            scenes[root] = []
        method_name = relative_path.parts[-3]
        scenes[root].append((method_name, path))

    results = {}
    for scene, paths in scenes.items():
        results[scene] = {"full": {}, "cropped": {}, "method_name": []}
        for (method_name, path) in paths:
            with open(path, 'r') as f:
                data = json.load(f)
                results[scene]["method_name"].append(method_name)

                for data_type in [key for key in results[scene].keys() if isinstance(results[scene][key], dict)]:
                    for key in data[data_type].keys():
                        if key not in results[scene][data_type].keys():
                            results[scene][data_type][key] = [data[data_type][key]]
                        else:
                            results[scene][data_type][key].append(data[data_type][key])

    # Generating tables
    tex = PyTex()

    columns = ["Scene", "Method name", "PSNR[dB]+", "PSNR$_M$[dB]+", "MAE[m]-", "MAE$_M$[m]-"]
    name_map = {"replace-nerf": "Our", "nerfacto-nerfiller": "Na\\\"{i}ve", "grid-prior-du": "Nerfiller"}
    digits = ([2, 2, 3, 2], [2, 2, 6, 6])

    fig_name = f"CV results"
    tex.table_cv(columns, results, 'llllll', digits, caption_title=fig_name, caption_text="Test", name_map=name_map)