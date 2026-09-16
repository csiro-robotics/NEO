import tyro
from argparse import ArgumentParser
from pathlib import Path

def main():
    parser = ArgumentParser()
    parser.add_argument("--neo_data_root", default="data/neo_dataset",
                        help="absolute path of output for dataset")
    parser.add_argument("--action", choices=["remove", "reconfigure", "assembly"])
    parser.add_argument("--scene_name", "-n", type=str, 
                        help="name of scene. Used in naming all related files")
    parser.add_argument("--object_name", type=str)
    parser.add_argument("--lerf_prompt", type=str)
    parser.add_argument("--model_types", type=str, choices=["neo"], nargs="+", default=["neo"],
                        help="Model type to edit")
    args=parser.parse_args()

    from robot_lerf.scripts.gen_grasp import GenGrasp

    scene_name = args.scene_name
    object_name = args.object_name
    data_path =  Path(args.action) / scene_name / object_name / "before"

    print(f"Data path: {data_path}")
    print(f"Lerf prompt: {args.lerf_prompt}")
    print(f"Neo Data Root: {args.neo_data_root}")

    gen_grasp = GenGrasp()
    gen_grasp.populate()

    gen_grasp.lerf_train_button_callback(neo_data_root=str(args.neo_data_root), lerf_dataset_path=str(data_path))
    gen_grasp.gen_grasp_button_callback(neo_data_root=str(args.neo_data_root), gen_grasp_text=args.lerf_prompt, model_types=args.model_types)
    
if __name__ == "__main__":
    tyro.cli(main)
