"""
emotion.py
----------
Main entry-point for the Emotion Recognition & Autism Detection project.

Usage:
    python emotion.py            -> Launches real-time webcam demo
    python emotion.py --train    -> Trains the CNN from scratch
    python emotion.py --evaluate -> Evaluates model on FER2013 test set
    python emotion.py --web      -> Starts Flask web dashboard
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Real-Time Emotion Recognition & Autism Detection\n"
            "Using CNN-Based Soft Computing Techniques\n"
            "---- B.Tech 2nd Year Project ----"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--train",    action="store_true",
        help="Train the emotion model on the RAF-DB dataset",
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Evaluate the saved model on the RAF-DB test split",
    )
    parser.add_argument(
        "--web",      action="store_true",
        help="Launch Flask web dashboard (optional)",
    )
    args = parser.parse_args()

    if args.train:
        print("[emotion] Starting model training ...")
        from config import USE_TRANSFER_LEARNING
        if USE_TRANSFER_LEARNING:
            from train import train_transfer_learning
            train_transfer_learning()
        else:
            from train import train_cnn
            train_cnn()

    elif args.evaluate:
        print("[emotion] Starting model evaluation ...")
        from evaluate import evaluate
        evaluate()

    elif args.web:
        print("[emotion] Starting Flask web app ...")
        from app import create_app
        app = create_app()
        app.run(host="0.0.0.0", port=5000, debug=False)

    else:
        # Default: real-time webcam demo
        print("[emotion] Starting real-time webcam demo ...")
        from realtime import run
        run()


if __name__ == "__main__":
    main()
