"""
Main script for loading an image, filtering segmentation by selected classes,
and displaying the result using the Laboratory class.
"""

from lib.Laboratory import Laboratory

def main():
    """
    Executes a sample segmentation workflow using the Laboratory wrapper.
    """
    laboratory = Laboratory()

    laboratory.show_categories() # -- optional

    laboratory.load_image("samples/first.png")
    laboratory.add_filter(15) # -- 15 == person
    laboratory.load()


if __name__ == "__main__":
    main()
