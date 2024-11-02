import torch
from data_handling import prepare_data_loader
from model import MultiLayerContrastiveLearning


def main():
    # Prepare data
    data_loader = prepare_data_loader("path/to/data.csv")

    # Initialize model
    model = MultiLayerContrastiveLearning()

    # Training loop here...


if __name__ == "__main__":
    main()
