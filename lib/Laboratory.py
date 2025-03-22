import torch
from torch import Tensor
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

class Laboratory:
    """
    A wrapper around the DeepLabV3_ResNet101 segmentation model for filtering
    and visualizing semantic classes on images.
    """

    def __init__(self):
        """
        Initializes the Laboratory instance, loads the model and sets default state.
        """
        self.repo: str = "pytorch/vision"
        self.model_name: str = "deeplabv3_resnet101"
        self.weights = DeepLabV3_ResNet101_Weights.DEFAULT
        self.weight_object: list[str] | None = self.weights.meta["categories"]  # Can be None (VOC)
        self.image: Image = None
        self.prediction: Tensor | None = None
        self.base_predication: Tensor | None = None
        self.filters: list[int] = []

        self._load_model()

    def _load_model(self) -> None:
        """
        Loads the pretrained model from Torch Hub and sets it to evaluation mode.
        """
        self.model = torch.hub.load(self.repo, self.model_name, weights=self.weights)
        self.model.eval()

    def show_categories(self) -> None:
        """
        Displays the list of class indices and their corresponding labels.
        """
        categories = self.weight_object
        if categories is None:
            print("No class labels available from the weight metadata.")
            return

        print("ID. CLASS_NAME")
        for i, name in enumerate(categories):
            print(f"{i:2}. {name}")

    def load_image(self, image_path: str) -> None:
        """
        Loads an image from the given path and converts it to RGB.

        Parameters:
            image_path (str): Path to the input image file.
        """
        try:
            self.image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Could not load image: {e}")

    def add_filter(self, *filters_id: int) -> None:
        """
        Adds one or more class IDs to the active filter list.

        Parameters:
            *filters_id (int): One or more class IDs to include in the output.
        """
        if self.weight_object is not None:
            max_classes = len(self.weight_object)
        else:
            max_classes = 21  # VOC-style fallback

        for filter_id in filters_id:
            if not (0 <= filter_id < max_classes):
                continue
            if filter_id not in self.filters:
                self.filters.append(filter_id)

    def _manage_image(self) -> None:
        """
        Preprocesses the loaded image for model inference.

        Returns:
            torch.Tensor: The preprocessed image tensor with batch dimension.
        """
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return preprocess(self.image).unsqueeze(0)

    def _infer(self) -> None:
        """
        Runs the segmentation model on the loaded image and stores predictions.
        """
        with torch.no_grad():
            output = self.model(self._manage_image())['out'][0]
        self.base_predication = output.argmax(0)
        self.prediction = self.base_predication.clone()

    def _process_filters(self) -> None:
        """
        Applies the active class filters to the prediction mask.
        """
        mask = torch.zeros_like(self.base_predication, dtype=torch.bool)

        if self.filters:
            for class_id in self.filters:
                mask |= (self.base_predication == class_id)
        else:
            # Default: show everything except background (class 0)
            mask = self.base_predication != 0

        self.prediction = self.base_predication.clone()
        self.prediction[~mask] = 255  # 255 = ignored

    def _show_image(self) -> None:
        """
        Displays the prediction mask with filtered classes using a colormap.
        """
        fig = plt.figure(figsize=(6, 6), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.imshow(self.prediction.numpy())
        plt.show()

    def load(self) -> None:
        """
        Performs the full pipeline: inference, filter application, and display.
        """
        self._infer()
        self._process_filters()
        self._show_image()
