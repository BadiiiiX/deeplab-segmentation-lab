"""
Main script for executing and managing segmentation tasks
using the Laboratory and Scenario classes.

Includes:
- Visual testing of segmentation on a single image
- Full scenario loading and saving to .npz
- Loading and inspecting a saved dataset
"""

import numpy
from lib.Laboratory import Laboratory
from lib.Scenario import Scenario


def show_image(image_path="samples/first.png"):
    """
    Performs segmentation on a single image using the Laboratory class.
    The mask will be filtered to keep only the 'person' class (ID: 15).

    Args:
        image_path (str): Path to the input image file.
    """
    laboratory = Laboratory()

    laboratory.show_categories()  # Optional: display available class labels

    laboratory.load_image(image_path)
    laboratory.add_filter(15)  # 15 == 'person' class in VOC
    laboratory.load()


def save_scenario(number=10, dataset_name="result_10"):
    """
    Loads and processes a complete scenario from DeepSense,
    and saves the output (masks + radio grids) to a compressed .npz file.

    Args:
        number (int): Scenario index to load (e.g., 10 for 'scenario_10').
        dataset_name (str): Output file name (without extension).
    """
    scenario = Scenario(number)
    scenario.save_numpy_dataset(dataset_name)


def load_numpy_dataset(dataset_name="result_10"):
    """
    Loads and prints the contents of a saved .npz dataset file.

    Args:
        dataset_name (str): File name (without extension) of the .npz dataset.
    """
    with numpy.load(f'{dataset_name}.npz') as data:
        print("Image IDs:")
        print(data["ids"])
        print("Radio Grids:")
        print(data["radios"])
        print("Masks:")
        print(data["masks"])


if __name__ == "__main__":
    # You can call one of the functions here for quick testing
    save_scenario()
    #show_image("./scenarios/10/camera/image_16.jpg")
