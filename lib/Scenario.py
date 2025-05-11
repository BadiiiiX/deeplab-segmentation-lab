from os import path, listdir
from typing import List, Any

import re

import numpy as np
from PIL import Image
from numpy import ndarray, dtype

from lib.Laboratory import Laboratory


def sort_list_dir(filename):
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else -1

class Scenario:
    """
    Loads a scenario consisting of synchronized image and radio data,
    and processes it using the Laboratory class.

    Note:
        This class is designed specifically for DeepSense scenarios.
    """

    scenario_path = "./scenarios/"
    camera_path = "camera"
    radio_path = "radio"

    def __init__(self, number: int):
        """
        Initializes the Scenario instance.

        Args:
            number (int): Scenario number to load from scenario_path.

        Raises:
            Exception: If the scenario path does not exist.
        """
        self.number = number
        self._check_scenario_number()
        self._init_laboratory()

        self.path = path.join(self.scenario_path, str(number))
        self.image_list = []
        self.radio_list = []
        self.result = []
        self._load_dataset()

    def _check_scenario_number(self):
        """
        Verifies that the specified scenario path exists.

        Raises:
            Exception: If the scenario directory does not exist.
        """
        path_to_check = path.join(self.scenario_path, str(self.number))
        if not path.exists(path_to_check):
            raise Exception(f"Invalid scenario number: {self.number} - path does not exist: {path_to_check}")

    def _init_laboratory(self) -> None:
        """
        Initializes the Laboratory instance for segmentation processing.
        """
        self.laboratory = Laboratory()

    def _get_laboratory_results(self, image_path: str) -> List[List[int]]:
        """
        Runs the segmentation pipeline on a given image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            List[List[int]]: Binary mask from segmentation output.
        """
        self.laboratory.load_image(image_path)
        self.laboratory.add_filter(15)  # class ID for 'person'
        return self.laboratory.load_in_array()

    def _crop_mask_to_multiple(self, binary_mask: List[List[int]], multiple=8):
        """
        Crops a binary mask so its dimensions are divisible by `multiple`.

        Args:
            binary_mask (List[List[int]]): Input binary mask.
            multiple (int): Target divisibility (default: 8).

        Returns:
            np.ndarray: Cropped binary mask.
        """
        mask = np.array(binary_mask)
        h, w = mask.shape
        h_new = h - (h % multiple)
        w_new = w - (w % multiple)
        return mask[:h_new, :w_new]

    def _load_power_grid(self, number: int):
        """
        Loads and reshapes the power values into an 8x8 grid.

        Args:
            number (int): Image number to match the corresponding power file.

        Returns:
            np.ndarray: 8x8 numpy array of power values.
        """
        def build_radio_name():
            return f"power_{number}.txt"

        radio_path = path.join(self.path, self.radio_path, build_radio_name())

        with open(radio_path, "r") as f:
            values = [float(line.strip()) for line in f if line.strip()]
        grid = np.array(values).reshape((8, 8))
        return grid

    def _get_data_from_image_number(self, number: int):
        """
        Loads and processes all data for a specific image index.

        Args:
            number (int): Index number of the image.

        Returns:
            dict: A dictionary containing:
                - "id": image index
                - "image": PIL.Image object
                - "radio": 8x8 np.ndarray of radio power values
                - "laboratory": binary segmentation mask (cropped)
        """
        def build_image_name():
            return f"image_{number}.jpg"

        image_path = path.join(self.path, self.camera_path, build_image_name())
        image_data = Image.open(image_path)

        radio_data = self._load_power_grid(number)
        laboratory = self._get_laboratory_results(image_path)
        laboratory_cropped = self._crop_mask_to_multiple(laboratory)

        return {
            "id": number,
            "radios": radio_data,
            "predictions": laboratory,
        }

    def _load_dataset(self):
        """
        Loads the full dataset (images + corresponding radio data) for the scenario.
        Processes and stores them in `self.result`.
        """
        def get_number_from_image(path_name):
            [_, number] = path_name.split("_")
            number = number.split(".")[0]
            return int(number)

        camera_dir = path.join(self.path, self.camera_path)

        if not path.exists(camera_dir):
            raise Exception("Missing camera directory in scenario path.")

        image_list = sorted(listdir(camera_dir), key=sort_list_dir)

        for c_path in image_list:
            print(f"Loading image: {c_path}...")
            data = self._get_data_from_image_number(get_number_from_image(c_path))
            self.result.append(data)

    def save_numpy_dataset(self, out_path: str = "dataset.npz"):
        """
        Saves the processed dataset to a compressed .npz file.

        Args:
            out_path (str): Destination path for the output .npz file.
        """
        masks = [entry["predictions"] for entry in self.result]
        radios = [np.rot90(entry["radios"], k=1) for entry in self.result]
        ids = [entry["id"] for entry in self.result]

        np.savez_compressed(out_path, ids=ids, masks=masks, radios=radios)
