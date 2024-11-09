import numpy as np
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageGrab, ImageOps
from model.neural_network import NeuralNetwork
from utils.save_load import save_weights_json, load_weights_json
from utils.data_loader import load_data
from sklearn.metrics import accuracy_score, precision_score, recall_score
from datetime import datetime
import matplotlib.pyplot as plt

class NeuralNetworkGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Neural Network GUI")

        self.canvas = tk.Canvas(self.master, width=200, height=200, bg='white')
        self.canvas.grid(row=0, column=0, columnspan=4)
        self.canvas.bind("<B1-Motion>", self.paint)

        # Epochs entry
        self.epochs_label = tk.Label(self.master, text="Epochs:")
        self.epochs_label.grid(row=1, column=0)
        self.epochs_entry = tk.Entry(self.master)
        self.epochs_entry.grid(row=1, column=1)
        self.epochs_entry.insert(0, '100')

        # Learning rate entry
        self.lr_label = tk.Label(self.master, text="Learning Rate:")
        self.lr_label.grid(row=2, column=0)
        self.lr_entry = tk.Entry(self.master)
        self.lr_entry.grid(row=2, column=1)
        self.lr_entry.insert(0, '0.001')

        # Batch size entry
        self.batch_size_label = tk.Label(self.master, text="Batch Size:")
        self.batch_size_label.grid(row=2, column=2)
        self.batch_size_entry = tk.Entry(self.master)
        self.batch_size_entry.grid(row=2, column=3)
        self.batch_size_entry.insert(0, '64')

        # Buttons
        self.train_button = tk.Button(self.master, text="Train", command=self.train_network)
        self.train_button.grid(row=3, column=0)
        
        self.show_graph_button = tk.Button(self.master, text="Show Graphs", command=self.show_graphs)
        self.show_graph_button.grid(row=3, column=1)

        self.save_button = tk.Button(self.master, text="Save Weights", command=self.save_weights)
        self.save_button.grid(row=3, column=2)

        self.load_button = tk.Button(self.master, text="Load Weights", command=self.load_weights)
        self.load_button.grid(row=3, column=3)

        self.recognize_button = tk.Button(self.master, text="Recognize", command=self.recognize)
        self.recognize_button.grid(row=3, column=4)

        self.clear_button = tk.Button(self.master, text="Clear Canvas", command=self.clear_canvas)
        self.clear_button.grid(row=3, column=5)

        self.prediction_text = tk.Text(self.master, height=12, width=30)
        self.prediction_text.grid(row=0, column=5, rowspan=4, padx=10, pady=10)

        self.metrics_label = tk.Label(self.master, text="")
        self.metrics_label.grid(row=5, column=0, columnspan=6)

        # Metric tracking variables
        self.epochs_list, self.losses, self.accuracies, self.precisions, self.recalls = [], [], [], [], []

        self.network = None

    def paint(self, event):
        x1, y1 = (event.x - 2), (event.y - 2)
        x2, y2 = (event.x + 2), (event.y + 2)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=5)

    def clear_canvas(self):
        self.canvas.delete("all")

    def recognize(self):
        if self.network is None:
            messagebox.showwarning("Warning", "Train or load a network first!")
            return

        # Capture the canvas
        x = self.master.winfo_rootx() + self.canvas.winfo_x()
        y = self.master.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()

        img = ImageGrab.grab().crop((x, y, x1, y1))
        img = img.resize((32, 32)).convert('L')
        img = ImageOps.invert(img)
        img_array = np.array(img).flatten() / 255.0

        predictions = self.network.forward(img_array.reshape(1, -1))[0]
        predicted_class_index = np.argmax(predictions)
        confidence = predictions[predicted_class_index] * 100
        classes = ['A', 'B', 'D', 'E', 'K', 'N', 'O', 'U', 'Y', 'Z']
        predicted_class = classes[predicted_class_index]
        self.prediction_text.delete(1.0, tk.END)
        self.prediction_text.insert(tk.END, f"Predicted Class: {predicted_class} with probability {confidence:.2f}%\n")
        messagebox.showinfo("Prediction", f"Predicted: {predicted_class} with {confidence:.2f}% confidence")

    def train_network(self):
        epochs = int(self.epochs_entry.get())
        learning_rate = float(self.lr_entry.get())
        batch_size = int(self.batch_size_entry.get())

        X_train, y_train, X_test, y_test = load_data()
        self.network = NeuralNetwork(input_size=1024, hidden_layers=[128, 64], output_size=10, learning_rate=learning_rate)

        for epoch in range(epochs):
            self.network.train(X_train, y_train, X_test, y_test, epochs=1, batch_size=batch_size)
            
            # Track metrics
            self.epochs_list.append(epoch + 1)
            loss = self.network.cross_entropy_loss(y_test, self.network.forward(X_test))
            accuracy = accuracy_score(y_test, self.network.predict(X_test))
            precision = precision_score(y_test, self.network.predict(X_test), average='weighted')
            recall = recall_score(y_test, self.network.predict(X_test), average='weighted')

            # Append metrics
            self.losses.append(loss)
            self.accuracies.append(accuracy)
            self.precisions.append(precision)
            self.recalls.append(recall)

            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy*100:.2f}%, Precision: {precision*100:.2f}%, Recall: {recall*100:.2f}%")

        messagebox.showinfo("Training Complete", f"Final Test Accuracy: {accuracy * 100:.2f}%")

    def show_graphs(self):
        if not self.epochs_list:
            messagebox.showwarning("Warning", "Train the network first!")
            return

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle("Training Metrics Over Epochs")

        # Plot accuracy
        axs[0, 0].plot(self.epochs_list, self.accuracies, label='Accuracy')
        axs[0, 0].set_title("Accuracy")
        axs[0, 0].set_xlabel("Epoch")
        axs[0, 0].set_ylabel("Accuracy")

        # Plot loss
        axs[0, 1].plot(self.epochs_list, self.losses, label='Loss', color='orange')
        axs[0, 1].set_title("Loss")
        axs[0, 1].set_xlabel("Epoch")
        axs[0, 1].set_ylabel("Loss")

        # Plot precision
        axs[1, 0].plot(self.epochs_list, self.precisions, label='Precision', color='green')
        axs[1, 0].set_title("Precision")
        axs[1, 0].set_xlabel("Epoch")
        axs[1, 0].set_ylabel("Precision")

        # Plot recall
        axs[1, 1].plot(self.epochs_list, self.recalls, label='Recall', color='red')
        axs[1, 1].set_title("Recall")
        axs[1, 1].set_xlabel("Epoch")
        axs[1, 1].set_ylabel("Recall")

        for ax in axs.flat:
            ax.legend()
            ax.grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def save_weights(self):
        if self.network is None:
            messagebox.showwarning("Warning", "Train or load a network first!")
            return
        save_weights_json(self.network)
        messagebox.showinfo("Info", "Weights saved successfully!")

    def load_weights(self):
        self.network = NeuralNetwork(input_size=1024, hidden_layers=[128, 64], output_size=10)
        load_weights_json(self.network)
        messagebox.showinfo("Info", "Weights loaded successfully!")
