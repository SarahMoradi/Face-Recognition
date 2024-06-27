import argparse
import pickle
from collections import Counter
from pathlib import Path

import face_recognition
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"

# Create directories if they don't already exist
Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)

parser = argparse.ArgumentParser(description="Recognize faces in an image")
parser.add_argument("--train", action="store_true", help="Train on input data")
parser.add_argument(
    "--validate", action="store_true", help="Validate trained model"
)
parser.add_argument(
    "--test", action="store_true", help="Test the model with an unknown image"
)
parser.add_argument(
    "-m",
    action="store",
    default="hog",
    choices=["hog", "cnn"],
    help="Which model to use for training: hog (CPU), cnn (GPU)",
)
parser.add_argument(
    "-f", action="store", help="Path to an image with an unknown face"
)
args = parser.parse_args()


def encode_known_faces(
    model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH
) -> None:

    names = []
    encodings = []

    for filepath in Path("training").glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

    name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)


def recognize_faces(
    image_location: str,
    model: str = "hog",
    encodings_location: Path = DEFAULT_ENCODINGS_PATH,
) -> None:

    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    input_image = face_recognition.load_image_file(image_location)

    input_face_locations = face_recognition.face_locations(
        input_image, model=model
    )
    input_face_encodings = face_recognition.face_encodings(
        input_image, input_face_locations
    )

    pillow_image = Image.fromarray(input_image)
    draw = ImageDraw.Draw(pillow_image)

    for bounding_box, unknown_encoding in zip(
        input_face_locations, input_face_encodings
    ):
        name = _recognize_face(unknown_encoding, loaded_encodings)
        if not name:
            name = "Unknown"
        _display_face(draw, bounding_box, name)

    del draw
    pillow_image.show()


def _recognize_face(unknown_encoding, loaded_encodings):

    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], unknown_encoding, tolerance=0.6
    )
    votes = Counter(
        name
        for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match
    )
    if votes:
        return votes.most_common(1)[0][0]


def _display_face(draw, bounding_box, name):

    top, right, bottom, left = bounding_box
    draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR)
    text_left, text_top, text_right, text_bottom = draw.textbbox(
        (left, bottom), name
    )
    draw.rectangle(
        ((text_left, text_top), (text_right, text_bottom)),
        fill=BOUNDING_BOX_COLOR,
        outline=BOUNDING_BOX_COLOR,
    )
    draw.text(
        (text_left, text_top),
        name,
        fill=TEXT_COLOR,
    )


def validate(model: str = "hog"):

    for filepath in Path("validation").rglob("*"):
        if filepath.is_file():
            recognize_faces(
                image_location=str(filepath.absolute()), model=model
            )


class FaceDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_paths = list(self.image_dir.glob("*/*.jpg"))
        self.label_names = sorted({path.parent.name for path in self.image_paths})
        self.label_to_idx = {label: idx for idx, label in enumerate(self.label_names)}
        self.labels = [self.label_to_idx[path.parent.name] for path in self.image_paths]

        print(f"Loaded {len(self.image_paths)} images from {image_dir}")
        print(f"Found labels: {self.label_names}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = face_recognition.load_image_file(image_path)
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


def plot_metrics(train_metrics, val_metrics, metric_name):
    plt.figure()
    plt.plot(train_metrics, label=f'Train {metric_name}')
    plt.plot(val_metrics, label=f'Validation {metric_name}')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    plt.title(f'Train and Validation {metric_name} over Epochs')
    plt.show()


if __name__ == "__main__":
    if args.train:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

        train_loader = DataLoader(FaceDataset("training", transform=transform), batch_size=32, shuffle=True)
        val_loader = DataLoader(FaceDataset("validation", transform=transform), batch_size=32, shuffle=False)

        if len(val_loader) == 0:
            raise ValueError("Validation dataset is empty. Please ensure there are validation samples.")

        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*128*3, 512),
            nn.ReLU(),
            nn.Linear(512, len(train_loader.dataset.label_names))
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        num_epochs = 10
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []

        for epoch in range(num_epochs):
            model.train()
            train_loss, correct, total = 0, 0, 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.clone().detach().to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            train_losses.append(train_loss / len(train_loader))
            train_accuracies.append(correct / total)

            model.eval()
            val_loss, correct, total = 0, 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.clone().detach().to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            val_losses.append(val_loss / len(val_loader))
            val_accuracies.append(correct / total)

        plot_metrics(train_losses, val_losses, 'Loss')
        plot_metrics(train_accuracies, val_accuracies, 'Accuracy')

    if args.validate:
        validate(model=args.m)
    if args.test:
        recognize_faces(image_location=args.f, model=args.m)
