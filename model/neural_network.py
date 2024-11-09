import numpy as np
from sklearn.metrics import f1_score, accuracy_score

class NeuralNetwork:
    def __init__(self, input_size=1024, hidden_layers=[128, 64], output_size=10,
                 learning_rate=0.001, l2_lambda=0.001, dropout_rate=0.5):
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate
        self.layer_sizes = [input_size] + hidden_layers + [output_size]
        self.weights = []
        self.biases = []

        # Инициализация весов методом He
        for i in range(len(self.layer_sizes) - 1):
            weight = np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * np.sqrt(2. / self.layer_sizes[i])
            bias = np.zeros(self.layer_sizes[i + 1])
            self.weights.append(weight)
            self.biases.append(bias)

        # Параметры оптимизатора Adam
        self.m_weights = [np.zeros_like(w) for w in self.weights]
        self.v_weights = [np.zeros_like(w) for w in self.weights]
        self.m_biases = [np.zeros_like(b) for b in self.biases]
        self.v_biases = [np.zeros_like(b) for b in self.biases]
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0  # Шаг времени для оптимизатора Adam

        self.best_weights = []
        self.best_biases = []
        self.best_accuracy = 0

    # Активационная функция ReLU и ее производная
    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    # Функция Softmax
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    # Функция потерь (кросс-энтропия)
    def cross_entropy_loss(self, y_true, y_pred):
        # Ensure y_true is one-hot encoded
        if y_true.ndim == 1:
            y_true = np.eye(y_pred.shape[1])[y_true]  # Convert to one-hot

        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

    # Прямое распространение
    def forward(self, X, training=True):
        self.activations = [X]
        self.Zs = []
        self.dropout_masks = []

        for i in range(len(self.weights)):
            Z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.Zs.append(Z)
            if i < len(self.weights) - 1:
                A = self.relu(Z)
                if training:
                    # Применяем Inverted Dropout
                    dropout_mask = (np.random.rand(*A.shape) < self.dropout_rate) / self.dropout_rate
                    A *= dropout_mask
                    self.dropout_masks.append(dropout_mask)
                else:
                    # Во время предсказания Dropout не используется
                    pass
                self.activations.append(A)
            else:
                A = self.softmax(Z)
                self.activations.append(A)
        return self.activations[-1]

    # Обратное распространение
    def backward(self, X, y):
        m = X.shape[0]
        y_encoded = np.zeros((m, self.weights[-1].shape[1]))
        y_encoded[np.arange(m), y] = 1

        dZ = self.activations[-1] - y_encoded
        dWs = []
        dbs = []

        for i in reversed(range(len(self.weights))):
            A_prev = self.activations[i]
            dW = (np.dot(A_prev.T, dZ) + self.l2_lambda * self.weights[i]) / m
            db = np.sum(dZ, axis=0) / m
            dWs.insert(0, dW)
            dbs.insert(0, db)

            if i > 0:
                dA_prev = np.dot(dZ, self.weights[i].T)
                # Применяем маску Dropout при обратном распространении
                if self.dropout_masks:
                    dA_prev *= self.dropout_masks[i - 1]
                dZ = dA_prev * self.relu_derivative(self.Zs[i - 1])

        # Обновление параметров с помощью оптимизатора Adam
        self.t += 1
        for i in range(len(self.weights)):
            # Обновление первого момента
            self.m_weights[i] = self.beta1 * self.m_weights[i] + (1 - self.beta1) * dWs[i]
            self.m_biases[i] = self.beta1 * self.m_biases[i] + (1 - self.beta1) * dbs[i]
            # Обновление второго момента
            self.v_weights[i] = self.beta2 * self.v_weights[i] + (1 - self.beta2) * (dWs[i] ** 2)
            self.v_biases[i] = self.beta2 * self.v_biases[i] + (1 - self.beta2) * (dbs[i] ** 2)
            # Коррекция смещения
            m_hat_w = self.m_weights[i] / (1 - self.beta1 ** self.t)
            m_hat_b = self.m_biases[i] / (1 - self.beta1 ** self.t)
            v_hat_w = self.v_weights[i] / (1 - self.beta2 ** self.t)
            v_hat_b = self.v_biases[i] / (1 - self.beta2 ** self.t)
            # Обновление весов и смещений
            self.weights[i] -= self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
            self.biases[i] -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

    # Оценка модели
    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        f1 = f1_score(y_test, predictions, average='weighted')
        accuracy = accuracy_score(y_test, predictions)
        return f1, accuracy


    # Обучение модели
    def train(self, X_train, y_train, X_test, y_test, epochs, batch_size):
        num_batches = int(np.ceil(X_train.shape[0] / batch_size))
        for epoch in range(epochs):
            permutation = np.random.permutation(X_train.shape[0])
            X_train_shuffled = X_train[permutation]
            y_train_shuffled = y_train[permutation]
            total_loss = 0

            for i in range(num_batches):
                start = i * batch_size
                end = start + batch_size
                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]
                output = self.forward(X_batch, training=True)
                self.backward(X_batch, y_batch)
                # Вычисление функции потерь для батча
                y_batch_encoded = np.zeros((y_batch.size, self.weights[-1].shape[1]))
                y_batch_encoded[np.arange(y_batch.size), y_batch] = 1
                loss = self.cross_entropy_loss(y_batch_encoded, output)
                total_loss += loss

            avg_loss = total_loss / num_batches
            f1, accuracy = self.evaluate(X_test, y_test)  # Распаковываем значения

            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_weights = [w.copy() for w in self.weights]
                self.best_biases = [b.copy() for b in self.biases]

            # Регулировка скорости обучения каждые 100 эпох
            if epoch % 100 == 99:
                self.learning_rate *= 0.98
                print(f"Снижена скорость обучения до {self.learning_rate}")

            print(f"Эпоха {epoch + 1}/{epochs}, Потеря: {avg_loss:.4f}, F1-Score: {f1 * 100:.2f}%, Точность: {accuracy * 100:.2f}%")

    # Предсказание
    def predict(self, X):
        output = self.forward(X, training=False)
        return np.argmax(output, axis=1)
