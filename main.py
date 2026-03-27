import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler


# 1. Data Preparation
def load_data():
    X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    y = y.reshape(-1, 1).astype(np.float32)
    return X, y


# 2. Model Architecture
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='tanh', input_shape=(2,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Force saturation (research trick)
    for layer in model.layers:
        if hasattr(layer, 'kernel'):
            layer.kernel.assign(
                tf.random.normal(layer.kernel.shape, mean=2.0, stddev=0.5)
            )

    return model


# 3. Training Function
def train_experiment(X, y, loss_type="BCE", epochs=200):
    model = build_model()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

    history = {'loss': [], 'grad_norm': []}

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            predictions = model(X)

            if loss_type == "BCE":
                loss = tf.keras.losses.binary_crossentropy(y, predictions)
            else:
                loss = tf.keras.losses.mse(y, predictions)

            avg_loss = tf.reduce_mean(loss)

        grads = tape.gradient(avg_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Track gradient norm (last layer)
        grad_norm = tf.linalg.norm(grads[-2]).numpy()

        history['loss'].append(avg_loss.numpy())
        history['grad_norm'].append(grad_norm)

    return history


# 4. Run Experiment
def run():
    X, y = load_data()

    bce_res = train_experiment(X, y, "BCE")
    mse_res = train_experiment(X, y, "MSE")

    # Plot
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(bce_res['loss'], label='Binary Cross-Entropy')
    plt.plot(mse_res['loss'], label='Mean Squared Error')
    plt.title('Loss Convergence')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Gradient Norm
    plt.subplot(1, 2, 2)
    plt.plot(bce_res['grad_norm'], label='BCE Gradient Norm')
    plt.plot(mse_res['grad_norm'], label='MSE Gradient Norm')
    plt.yscale('log')
    plt.title('Gradient Magnitude (Saturation Test)')
    plt.xlabel('Epochs')
    plt.ylabel('Gradient Norm')
    plt.legend()

    plt.tight_layout()
    plt.savefig("results/experiment.png")
    plt.show()


if __name__ == "__main__":
    run()