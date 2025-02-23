import json
from pathlib import Path

from nerfstudio.utils.rich_utils import CONSOLE


def validate_transforms(path_to_transforms: Path) -> None:
    """Validate the transforms.json file for the correct
    format. If the file has the wrong format, correct it.
    """

    # Load transforms
    transforms = None
    with open(path_to_transforms / "transforms.json", 'r') as f:
        transforms = json.load(f)

    # Check if transfomrs have proper format
    assert transforms is not None, "Transforms are not loaded properly."

    if "fl_x" in transforms:
        CONSOLE.print("[yellow]transforms.json has wrong format. Correcting the file")

        for frame in transforms["frames"]:
            frame["w"] = transforms["w"]
            frame["h"] = transforms["h"]
            frame["fl_x"] = transforms["fl_x"]
            frame["fl_y"] = transforms["fl_y"]
            frame["cx"] = transforms["cx"]
            frame["cy"] = transforms["cy"]
            frame["k1"] = transforms["k1"]
            frame["k2"] = transforms["k2"]
            frame["p1"] = transforms["p1"]
            frame["p2"] = transforms["p2"]

        del transforms["w"]
        del transforms["h"]
        del transforms["fl_x"]
        del transforms["fl_y"]
        del transforms["cx"]
        del transforms["cy"]
        del transforms["k1"]
        del transforms["k2"]
        del transforms["p1"]
        del transforms["p2"]

        with open(path_to_transforms / "transforms.json", 'w') as f:
            json.dump(transforms, f, indent=4)