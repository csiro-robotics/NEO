import subprocess
ROOT_BAG = "data/bowen-neo/raw_bagfiles"
ROOT_OUTPUT = "data/bowen-neo/neo_dataset"
# SCENE_NAMES = ["bob_the_builder", "cast_away", "fast_and_furious", "hells_kitchen", "indiana_jones", "mustard_bottle", "orange", "red_robot", "spam", "the_silence_of_the_lambs", "tomato_soup_can"]
SCENE_NAMES = ["indiana_jones", "mustard_bottle", "orange", "red_robot", "spam", "the_silence_of_the_lambs", "tomato_soup_can"]
# ACTION_SETS = {"remove":SCENE_NAMES, "translate": ["mustard_bottle", "orange", "red_robot", "spam", "tomato_soup_can"], "translate_and_rotate": ["mustard_bottle", "orange", "red_robot", "spam", "tomato_soup_can"]}
ACTION_SETS = {"translate": ["mustard_bottle", "orange", "red_robot", "spam", "tomato_soup_can"], "translate_and_rotate": ["mustard_bottle", "orange", "red_robot", "spam", "tomato_soup_can"]}
# SCENE_TYPES = {"single": ["mustard_bottle", "orange", "red_robot", "spam", "tomato_soup_can"], "multiple_one": ["bob_the_builder", "cast_away", "fast_and_furious", "hells_kitchen", "the_silence_of_the_lambs"], "multiple_sequential": ["indiana_jones"]}
SCENE_TYPES = {"mustard_bottle": "single", "orange": "single", "red_robot": "single", "spam": "single", "tomato_soup_can": "single", "bob_the_builder": "multiple_one", "cast_away": "multiple_one", "fast_and_furious": "multiple_one", "hells_kitchen": "multiple_one", "the_silence_of_the_lambs": "multiple_one", "indiana_jones": "multiple_sequential"}
SCENE_OBJECTS = {"bob_the_builder": ["tape_measure"], "cast_away": ["yellow_submarine"], "fast_and_furious": ["thomas_the_tank_engine"], "hells_kitchen": ["pear"], "the_silence_of_the_lambs": ["pink_sheep"], "indiana_jones": ["elephant_rider", "tiger", "jelly", "yellow_car", "green_block"], "mustard_bottle": ["mustard_bottle"], "orange": ["orange"], "red_robot": ["red_robot"], "spam": ["spam"], "tomato_soup_can": ["tomato_soup_can"]}

for action, scene_names in ACTION_SETS.items():
    for scene_name in scene_names:
        if SCENE_TYPES[scene_name] == "multiple_sequential":
            for idx, object in enumerate(SCENE_OBJECTS[scene_name]):
                subprocess.run(["python", "data_processing/process_bags.py",
                               "--root_bag", f"{ROOT_BAG}", "--root_output", f"{ROOT_OUTPUT}",
                               "--scene_name", f"{scene_name}",
                               "--action", f"{action}",
                               "--scene_type", f"{SCENE_TYPES[scene_name]}",
                               "--sequence_num", f"{idx+1}", 
                               "--object_name", f"{object}"])
        elif SCENE_TYPES[scene_name] == "single":
            subprocess.run(["python", "data_processing/process_bags.py",
                            "--root_bag", f"{ROOT_BAG}", "--root_output", f"{ROOT_OUTPUT}",
                            "--scene_name", f"{scene_name}",
                            "--action", f"{action}",
                            "--scene_type", f"{SCENE_TYPES[scene_name]}"])
        else:
            subprocess.run(["python", "data_processing/process_bags.py",
                            "--root_bag", f"{ROOT_BAG}", "--root_output", f"{ROOT_OUTPUT}",
                            "--scene_name", f"{scene_name}",
                            "--action", f"{action}",
                            "--scene_type", f"{SCENE_TYPES[scene_name]}",
                            "--object_name", f"{SCENE_OBJECTS[scene_name][0]}"])
