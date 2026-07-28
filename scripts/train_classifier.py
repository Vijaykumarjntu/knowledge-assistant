import argparse
from pathlib import Path

from app.services.classifier import CATEGORY_LABELS, ClassifierService


def build_training_dataset():
    texts = []
    labels = []

    examples = {
        "Artificial Intelligence": [
            "Research on autonomous reasoning and intelligent agents for planning and decision-making.",
            "A framework for symbolic reasoning and knowledge graphs in AI systems.",
            "A study on neural symbolic integration for artificial intelligence solutions.",
        ],
        "Machine Learning": [
            "Supervised learning methods for regression and classification tasks.",
            "A deep learning training pipeline using convolutional neural networks.",
            "Model evaluation strategies for hyperparameter search and cross-validation.",
        ],
        "Computer Vision": [
            "Object detection and image segmentation for autonomous vehicles.",
            "A paper on visual feature extraction and scene understanding.",
            "Techniques for real-time video analysis and camera-based perception.",
        ],
        "Natural Language Processing": [
            "Text classification, named entity recognition, and transformer-based language models.",
            "A study of sentiment analysis and question answering over documents.",
            "Language understanding and sequence-to-sequence modelling for NLP applications.",
        ],
        "Robotics": [
            "Control algorithms and motion planning for industrial robotic arms.",
            "A survey of navigation systems and autonomous mobile robots.",
            "Robot perception using lidar and sensor fusion methods.",
        ],
        "Cyber Security": [
            "Threat detection systems and intrusion prevention for enterprise networks.",
            "A review of secure software development and vulnerability management.",
            "Encryption, access control, and identity management for cyber security.",
        ],
        "Cloud Computing": [
            "Scalable microservices deployment and serverless architecture in the cloud.",
            "A study on cloud infrastructure, container orchestration, and hybrid cloud solutions.",
            "Cloud-native data pipelines and distributed storage best practices.",
        ],
    }

    for label, examples_list in examples.items():
        for text in examples_list:
            texts.append(text)
            labels.append(label)

    return {"texts": texts, "labels": labels}


def main():
    parser = argparse.ArgumentParser(description="Train the document classifier.")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs.")
    args = parser.parse_args()
    data = build_training_dataset()
    classifier = ClassifierService()
    classifier.train(data["texts"], data["labels"], epochs=args.epochs)
    print("Classifier training completed. Artifacts stored in storage/classifier.")


if __name__ == "__main__":
    main()
