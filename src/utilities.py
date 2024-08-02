import json
import os


def get_execution_time(start_time, end_time):

    elapsed_time = end_time - start_time

    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60

    return (
        f"Execution time: {hours} hours, {minutes} minutes, and {seconds:.2f} seconds"
    )


def read_config(config_file):
    with open(config_file, "r") as json_file:
        config_dict = json.load(json_file)

    return config_dict


def create_directories(path):

    path_components = path.split(os.sep)
    current_path = "" if path.startswith(os.sep) else "."

    for component in path_components:
        if component:
            current_path = os.path.join(current_path, component)
            if not os.path.exists(current_path):
                os.makedirs(current_path)
                print(f"Directory created: {current_path}", flush=True)

            else:
                print(f"Directory already exists: {current_path}", flush=True)
