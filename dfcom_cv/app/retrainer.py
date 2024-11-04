class ModelReloader:
    def __init__(self, cnn_trainer, transfer_learning_trainer, save_dir="models"):
        """
        Initialize the ModelReloader with CNN and transfer learning trainers.

        Args:
            cnn_trainer (CNNTrainer): Instance of the CNNTrainer.
            transfer_learning_trainer (TransferLearningTrainer): Instance of the Class.
            save_dir (str): Directory where the models will be saved.
        """
        self.cnn_trainer = cnn_trainer
        self.transfer_learning_trainer = transfer_learning_trainer
        self.save_dir = save_dir

    def retrain_cnn(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        """
        Retrain the CNN model.

        Args:
            X_train (np.array): Training images.
            y_train (np.array): Training labels.
            X_val (np.array): Validation images.
            y_val (np.array): Validation labels.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.

        Returns:
            History: Training history object.
        """
        print("Retraining CNN model...")
        history = self.cnn_trainer.train(X_train, y_train, X_val, y_val,
                                         epochs=epochs,
                                         batch_size=batch_size)
        self.cnn_trainer.save_model(save_dir=self.save_dir)
        return history

    def retrain_transfer_learning(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        """
        Retrain the transfer learning model.

        Args:
            X_train (np.array): Training images.
            y_train (np.array): Training labels.
            X_val (np.array): Validation images.
            y_val (np.array): Validation labels.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.

        Returns:
            History: Training history object.
        """
        print("Retraining Transfer Learning model...")
        history = self.transfer_learning_trainer.train(X_train, y_train, X_val, y_val,
                                                       epochs=epochs,
                                                       batch_size=batch_size)
        self.transfer_learning_trainer.save_model(save_dir=self.save_dir)
        return history
