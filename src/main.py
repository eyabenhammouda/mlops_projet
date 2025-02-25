
import argparse
import os
import model as ml_model
import tensorflow as tf
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="ML Project Pipeline")
    parser.add_argument('--prepare', action='store_true', help='Prepare the data')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--save', type=str, help='Save the model to a file')
    parser.add_argument('--train_path', type=str, required=True, help='Path to the training data')
    parser.add_argument('--test_path', type=str, required=True, help='Path to the testing data')
    
    args = parser.parse_args()

    # Configure TensorBoard logging
    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(log_dir)

    if args.prepare:
        X_train, X_test, y_train, y_test = ml_model.prepare_data(args.train_path, args.test_path)
        print("Data prepared successfully.")

    if args.train:
        X_train, X_test, y_train, y_test = ml_model.prepare_data(args.train_path, args.test_path)
        model = ml_model.train_model(X_train, y_train)
        best_model = ml_model.optimize_hyperparameters(X_train, y_train)
        
        # Log metrics to TensorBoard
        with file_writer.as_default():
            accuracy, precision, recall, f1 = ml_model.evaluate_model(best_model, X_test, y_test)
            tf.summary.scalar('accuracy', accuracy, step=1)
            tf.summary.scalar('precision', precision, step=1)
            tf.summary.scalar('recall', recall, step=1)
            tf.summary.scalar('f1_score', f1, step=1)
        
        # Save the model
        ml_model.save_model(best_model, "models/best_model.pkl")
        
        print("Model trained and hyperparameters optimized successfully.")

    if args.evaluate:
        X_train, X_test, y_train, y_test = ml_model.prepare_data(args.train_path, args.test_path)
        model = ml_model.load_model(args.save)
        accuracy, precision, recall, f1 = ml_model.evaluate_model(model, X_test, y_test)
        print(f'Model accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')

    if args.save:
        X_train, X_test, y_train, y_test = ml_model.prepare_data(args.train_path, args.test_path)
        model = ml_model.train_model(X_train, y_train)
        best_model = ml_model.optimize_hyperparameters(X_train, y_train)
        ml_model.save_model(best_model, args.save)
        print(f'Model saved to {args.save}')

if __name__ == "__main__":
    main()

