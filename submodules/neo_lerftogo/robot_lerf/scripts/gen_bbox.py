import tyro
from argparse import ArgumentParser
from pathlib import Path

def main():
    parser = ArgumentParser()
    parser.add_argument("--root_input", default="data/neo_dataset",
                        help="absolute path of output for dataset")
    parser.add_argument("--data_type", type=str, choices=["real", "simulated"],
                    help="specify if data is real or simulated")
    parser.add_argument("--action", choices=["remove", "translate", 
                                        "rotate", "translate_and_rotate",
                                        "mix", "assembly"])
    parser.add_argument("--scene_type", choices=["single", "multiple_one", 
                                                "multiple_sequential", 
                                                "multiple_multiaction"])
    parser.add_argument("--scene_name", "-n", type=str, 
                        help="name of scene. Used in naming all related files")
    parser.add_argument("--object_name", type=str)
    parser.add_argument("--lerf_prompt", type=str)
    parser.add_argument("--model_types", type=str, choices=["neo"], nargs="+", default=["neo"],
                        help="Model type to edit")
    args=parser.parse_args()

    from robot_lerf.scripts.gen_grasp import GenGrasp

    scene_name = args.scene_name
    if args.scene_type == "single":
        object_name = args.object_name
    elif not args.object_name:
        raise ValueError("Cannot have action on multiple object scene without defining object being acted upon")
    else:
        object_name = "-".join([obj for obj in args.object_name])
    data_path = Path(args.data_type) / args.action / args.scene_type / scene_name / object_name / "before"

    print(f"Data path: {data_path}")
    print(f"Lerf prompt: {args.lerf_prompt}")
    print(f"Root input: {args.root_input}")

    gen_grasp = GenGrasp()
    gen_grasp.populate()

    gen_grasp.lerf_train_button_callback(root_input=str(args.root_input), lerf_dataset_path=str(data_path))
    gen_grasp.gen_grasp_button_callback(root_input=str(args.root_input), gen_grasp_text=args.lerf_prompt, model_types=args.model_types)
    
if __name__ == "__main__":
    tyro.cli(main)
