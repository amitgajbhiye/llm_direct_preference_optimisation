import json


def get_execution_time(start_time, end_time):

    elapsed_time = end_time - start_time

    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60

    # print(
    #     f"Execution time: {hours} hours, {minutes} minutes, and {seconds:.2f} seconds",
    #     flush=True,
    # )

    return (
        f"Execution time: {hours} hours, {minutes} minutes, and {seconds:.2f} seconds"
    )


def read_config(config_file):
    with open(config_file, "r") as json_file:
        config_dict = json.load(json_file)

    return config_dict
