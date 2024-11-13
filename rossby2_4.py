import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, clone_model
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Conv2D,
    LeakyReLU,
    BatchNormalization,
    Reshape,
    Flatten,
    Input,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import io

# Diagnostic Information
import sys

st.image("rossby_gif.gif", use_column_width=True)
st.title("Rossby Wave Deep Learning Prediction App")

st.markdown("### **Shammunul Islam** 👋")

# Add a horizontal separator for visual clarity
st.markdown("---")
# Add your name with custom styling using HTML and CSS
# st.markdown(
#     """
#     <div style="text-align: left; margin-top: -10px;">
#         <h3 style="color: #4B8BBE; font-family: 'Arial', sans-serif;">
#             <b>Shammunul Islam</b>
#         </h3>
#     </div>
#     """,
#     unsafe_allow_html=True
# )

# Function to generate Rossby wave simulation data
def generate_rossby_wave_data(Lx, Ly, Nx, Ny, beta, dt, T, k_x, k_y, sigma):
    dx, dy = Lx / Nx, Ly / Ny
    steps = int(T / dt)
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)
    psi_sin = np.sin(k_x * X) * np.sin(k_y * Y)
    x0, y0 = Lx / 2, Ly / 2
    psi_gaussian = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma ** 2))
    psi = psi_gaussian.copy()

    def laplacian_2d(psi, dx, dy):
        lap_psi_x = (np.roll(psi, -1, axis=0) - 2 * psi + np.roll(psi, 1, axis=0)) / dx ** 2
        lap_psi_y = (np.roll(psi, -1, axis=1) - 2 * psi + np.roll(psi, 1, axis=1)) / dy ** 2
        return lap_psi_x + lap_psi_y

    def time_step(psi, beta, dx, dt):
        psi_new = psi - dt * beta * (np.roll(psi, -1, axis=0) - np.roll(psi, 1, axis=0)) / (2 * dx)
        return psi_new

    data = []
    for n in range(steps):
        psi = time_step(psi, beta, dx, dt)
        data.append(psi.copy())
    return np.array(data), psi_gaussian  # Return initial wave

from scipy.ndimage import zoom
from sklearn.preprocessing import StandardScaler

# Data preparation for LSTM
def prepare_data_lstm(data, sequence_length, Nx, Ny):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i+sequence_length])
        targets.append(data[i+sequence_length])
    sequences = np.array(sequences)
    targets = np.array(targets)
    # Reshape for LSTM
    sequences = sequences.reshape(sequences.shape[0], sequence_length, Nx * Ny)
    targets = targets.reshape(targets.shape[0], Nx * Ny)
    return sequences, targets

# Data preparation for CNN
def prepare_data_cnn(data, Nx, Ny):
    X = data[:-1]
    y = data[1:]
    # Reshape for CNN
    X = X.reshape(X.shape[0], Nx, Ny, 1)
    y = y.reshape(y.shape[0], Nx, Ny, 1)  # Ensure target matches model output
    return X, y

# Data preparation for PINNs
def prepare_data_pinn(data, Lx, Ly, T, beta, num_domain=2540, num_boundary=80, num_initial=160):
    def rossby_wave_pde(x, y):
        psi = y
        dpsi_t = dde.grad.jacobian(psi, x, i=0, j=2)
        dpsi_xx = dde.grad.hessian(psi, x, i=0, j=0)
        dpsi_yy = dde.grad.hessian(psi, x, i=1, j=1)
        laplacian = dpsi_xx + dpsi_yy
        return dpsi_t + beta * dde.grad.jacobian(psi, x, i=0, j=0) - laplacian

    geom = dde.geometry.geometry_2d.Rectangle([0, 0], [Lx, Ly])
    timedomain = dde.geometry.TimeDomain(0, T)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    data_pinn = dde.data.TimePDE(
        geomtime,
        rossby_wave_pde,
        [],
        num_domain=num_domain,
        num_boundary=num_boundary,
        num_initial=num_initial,
    )
    return data_pinn

# Custom callback to display training progress in Streamlit
class StreamlitCallback(tf.keras.callbacks.Callback):
    def __init__(self, iterations):
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        self.iterations = iterations

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get("loss")
        self.status_text.text(
            f"Iteration {epoch+1}/{self.iterations}, Loss: {loss:.6f}"
        )
        self.progress_bar.progress((epoch + 1) / self.iterations)

# Add the description with mathematical expressions
st.markdown(
    """
### Two-Dimensional Rossby Wave Equation

The two-dimensional version of the Barotropic Rossby wave equation is [Source](https://github.com/CompPhysics/ComputationalPhysics/blob/master/doc/Projects/2020/Project5/WaveEquation/ipynb/WaveEquation.ipynb):
"""
)

st.latex(r"""
\frac{\partial}{\partial t} \nabla_H^2 \psi + \beta \frac{\partial \psi}{\partial x} = 0
""")

st.markdown(r"""
Where 

$$\nabla_H^2 \psi = \partial_{xx} \psi + \partial_{yy} \psi$$

represents the Laplacian in two dimensions.

We can use finite difference methods to solve the Rossby wave equation numerically. The process involves calculating spatial derivatives. Spatial derivatives are calculated using second derivatives in both x and y directions:
""")

st.latex(r"""
\frac{\partial^2 \psi}{\partial x^2} \approx \frac{\psi_{i+1,j} - 2 \psi_{i,j} + \psi_{i-1,j}}{\Delta x^2}
""")

st.latex(r"""
\frac{\partial^2 \psi}{\partial y^2} \approx \frac{\psi_{i,j+1} - 2 \psi_{i,j} + \psi_{i,j-1}}{\Delta y^2}
""")

st.markdown("""
**Boundary Conditions:**

We used both sinusoidal and Gaussian initial conditions. In 2D:
""")

st.markdown("- **Sinusoidal:**")
st.latex(r"\psi(x, y) = \sin(4\pi x) \sin(4\pi y)")

st.markdown("- **Gaussian:**")
st.latex(r"\psi(x, y) = \exp\left( - \frac{(x - x_{0})^{2} + (y - y_{0})^{2}}{2\sigma^{2}} \right)")

# Sidebar settings

# Added unique key to avoid duplicate element IDs
model_type = st.sidebar.selectbox(
    "Choose Model Type",
    ["LSTM", "CNN", "GAN", "PINNs"],
    key="model_type_selectbox",
)

# Simulation parameters for wave generation
st.sidebar.header("Simulation Parameters")

# Reduced default values for Nx and Ny to decrease data size
Nx = st.sidebar.slider("Grid points in x (Nx)", 20, 100, 50, step=10, key="Nx_slider")
Ny = st.sidebar.slider("Grid points in y (Ny)", 20, 100, 50, step=10, key="Ny_slider")

Lx = st.sidebar.slider("Domain length (Lx)", 0.5, 4.0, 1.0, step=0.1, key="Lx_slider")
Ly = st.sidebar.slider("Domain length (Ly)", 0.5, 4.0, 1.0, step=0.1, key="Ly_slider")
beta = st.sidebar.slider("Rossby parameter (β)", 0.1, 3.0, 1.0, step=0.1, key="beta_slider")
k_x = st.sidebar.slider(
    "Wave number in x (kx)",
    float(2 * np.pi),
    float(8 * np.pi),
    float(4 * np.pi),
    key="kx_slider",
)
k_y = st.sidebar.slider(
    "Wave number in y (ky)",
    float(2 * np.pi),
    float(8 * np.pi),
    float(4 * np.pi),
    key="ky_slider",
)
sigma = st.sidebar.slider("Gaussian width (σ)", 0.05, 1.0, 0.1, step=0.05, key="sigma_slider")
T = st.sidebar.slider(
    "Total simulation time (T)", 5.0, 50.0, 10.0, step=5.0, key="T_slider"
)

# Increased default dt to reduce number of steps
dt = st.sidebar.slider("Time step (dt)", 0.001, 0.01, 0.005, step=0.001, key="dt_slider")

# Add button to generate Rossby wave data
if st.button("Generate Rossby Wave", key="generate_button"):
    # Generate simulation data
    data, initial_wave = generate_rossby_wave_data(
        Lx, Ly, Nx, Ny, beta, dt, T, k_x, k_y, sigma
    )

    # Normalize data
    data_min = data.min()
    data_max = data.max()
    data_norm = (data - data_min) / (data_max - data_min)

    # Store data in session state
    st.session_state["data"] = data_norm
    st.session_state["data_full"] = data  # Store full data
    st.session_state["data_min"] = data_min
    st.session_state["data_max"] = data_max
    st.session_state["Nx"] = Nx
    st.session_state["Ny"] = Ny
    st.session_state["initial_wave"] = initial_wave  # Store initial wave
    st.session_state["beta"] = beta
    st.session_state["T"] = T
    st.session_state["Lx"] = Lx
    st.session_state["Ly"] = Ly

    st.success("Rossby wave data generated.")

# Always display the initial wave if generated
if "initial_wave" in st.session_state:
    st.subheader("Initial Rossby Wave State")
    initial_wave = st.session_state["initial_wave"]
    fig, ax = plt.subplots()
    cax = ax.imshow(initial_wave, cmap="RdBu_r", origin="lower")
    ax.set_title("Initial Rossby Wave State")
    fig.colorbar(cax, ax=ax, label="Streamfunction ψ")
    st.pyplot(fig)

# Model hyperparameters based on user selection
if model_type == "LSTM":
    st.sidebar.header("LSTM Hyperparameters")
    sequence_length = st.sidebar.slider("Sequence Length", 5, 20, 10, key="sequence_length_slider")
    lstm_units = st.sidebar.slider("LSTM Units", 32, 128, 64, key="lstm_units_slider")
    pinn_iterations = st.sidebar.slider("Iterations", 10, 10000, 10, key="pinn_iterations_slider")  # Renamed
    batch_size = st.sidebar.slider("Batch Size", 8, 32, 16, key="lstm_batch_size_slider")
    max_samples = st.sidebar.slider("Max Samples", 1000, 5000, 2000, step=500, key="max_samples_slider")

elif model_type == "CNN":
    st.sidebar.header("CNN Hyperparameters")
    filters = st.sidebar.slider("Number of Filters", 8, 64, 32, key="filters_slider")
    kernel_size = st.sidebar.slider("Kernel Size", 3, 7, 3, key="kernel_size_slider")
    pinn_iterations = st.sidebar.slider("Iterations", 10, 10000, 10, key="pinn_iterations_slider")  # Renamed
    batch_size = st.sidebar.slider("Batch Size", 8, 32, 16, key="cnn_batch_size_slider")

elif model_type == "GAN":
    st.sidebar.header("GAN Hyperparameters")
    latent_dim = st.sidebar.slider("Latent Dimension", 50, 200, 100, key="latent_dim_slider")
    generator_units = st.sidebar.slider("Generator Units", 64, 256, 128, key="generator_units_slider")
    discriminator_units = st.sidebar.slider(
        "Discriminator Units", 64, 256, 128, key="discriminator_units_slider"
    )
    pinn_iterations = st.sidebar.slider("Iterations", 100, 10000, 100, key="gan_iterations_slider")  # Renamed
    batch_size = st.sidebar.slider("Batch Size", 8, 64, 32, key="gan_batch_size_slider")

elif model_type == "PINNs":
    st.sidebar.header("PINNs Hyperparameters")
    pinn_iterations = st.sidebar.slider("Iterations", 100, 10000, 100, key="pinn_iterations_slider")  # Updated
    learning_rate = st.sidebar.slider(
        "Learning Rate", 1e-5, 1e-2, 1e-3, format="%.5f", key="learning_rate_slider"
    )

# Run the model
if st.button("Run the model", key="run_model_button"):
    # Check if data has been generated
    if "data" in st.session_state:
        data = st.session_state["data"]
        data_min = st.session_state["data_min"]
        data_max = st.session_state["data_max"]
        Nx = st.session_state["Nx"]
        Ny = st.session_state["Ny"]
        beta = st.session_state["beta"]
        T = st.session_state["T"]
        Lx = st.session_state["Lx"]
        Ly = st.session_state["Ly"]

        st.subheader("Model Configuration")
        st.write(f"**Model Type**: {model_type}")

        if model_type == "LSTM":
            # Ensure pinn_iterations is defined
            if 'pinn_iterations' not in locals():
                pinn_iterations = st.session_state.get('pinn_iterations', 5000)
            st.write(f"**Iterations**: {pinn_iterations}")  # Moved inside the block
            st.write(f"**Sequence Length**: {sequence_length}")
            st.write(f"**LSTM Units**: {lstm_units}")
            st.write(f"**Batch Size**: {batch_size}")

            # Prepare data for LSTM
            X_train_lstm, y_train_lstm = prepare_data_lstm(data, sequence_length, Nx, Ny)

            # Clear previous models to prevent memory issues
            tf.keras.backend.clear_session()

            # Build LSTM model
            model = Sequential([
                LSTM(lstm_units, input_shape=(sequence_length, Nx * Ny)),
                Dense(Nx * Ny)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')

            # Display model summary
            string_io = io.StringIO()
            model.summary(print_fn=lambda x: string_io.write(x + '\n'))
            summary_string = string_io.getvalue()
            st.subheader("Model Summary")
            st.text(summary_string)

            # Define callbacks
            early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

            # Train model with StreamlitCallback and EarlyStopping
            st.subheader("Training Progress")
            history = model.fit(
                X_train_lstm,
                y_train_lstm,
                epochs=pinn_iterations,  # Use iterations instead of epochs
                batch_size=batch_size,
                callbacks=[StreamlitCallback(pinn_iterations), early_stopping],
                verbose=0
            )

            st.success("LSTM Model Trained")

            # Store the trained model and data in session state
            st.session_state['model'] = model
            st.session_state['model_type'] = model_type
            st.session_state['sequence_length'] = sequence_length
            st.session_state['X_train_lstm'] = X_train_lstm
            st.session_state['y_train_lstm'] = y_train_lstm  # Store y_train for comparison

            # Display training loss
            final_loss = history.history['loss'][-1]
            st.subheader("Training Loss")
            st.write(f"Final Training Loss: {final_loss:.6f}")

            # Optional: Plot training loss
            fig_loss, ax_loss = plt.subplots()
            ax_loss.plot(history.history['loss'], label='Training Loss')
            if 'val_loss' in history.history:
                ax_loss.plot(history.history['val_loss'], label='Validation Loss')
            ax_loss.set_title('Loss Curves')
            ax_loss.set_xlabel('Iteration')
            ax_loss.set_ylabel('Loss')
            ax_loss.legend()
            st.pyplot(fig_loss)

        elif model_type == "CNN":
            # Ensure pinn_iterations is defined
            if 'pinn_iterations' not in locals():
                pinn_iterations = st.session_state.get('pinn_iterations', 5000)
            st.write(f"**Iterations**: {pinn_iterations}")  # Moved inside the block
            st.write(f"**Filters**: {filters}")
            st.write(f"**Kernel Size**: {kernel_size}")
            st.write(f"**Batch Size**: {batch_size}")

            # Prepare data for CNN
            X_train_cnn, y_train_cnn = prepare_data_cnn(data, Nx, Ny)

            # Clear previous models to prevent memory issues
            tf.keras.backend.clear_session()

            # Build CNN model with updated architecture
            model = Sequential([
                Conv2D(filters, (kernel_size, kernel_size), activation='relu', padding='same', input_shape=(Nx, Ny, 1)),
                Conv2D(filters, (kernel_size, kernel_size), activation='relu', padding='same'),
                Conv2D(1, (kernel_size, kernel_size), activation='linear', padding='same')  # Changed layer
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')

            # Display model summary
            string_io = io.StringIO()
            model.summary(print_fn=lambda x: string_io.write(x + '\n'))
            summary_string = string_io.getvalue()
            st.subheader("Model Summary")
            st.text(summary_string)

            # Define callbacks
            early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

            # Train model with StreamlitCallback and EarlyStopping
            st.subheader("Training Progress")
            history = model.fit(
                X_train_cnn,
                y_train_cnn,
                epochs=pinn_iterations,  # Use iterations instead of epochs
                batch_size=batch_size,
                callbacks=[StreamlitCallback(pinn_iterations), early_stopping],
                verbose=0
            )

            st.success("CNN Model Trained")

            # Store the trained model and data in session state
            st.session_state['model'] = model
            st.session_state['model_type'] = model_type
            st.session_state['X_train_cnn'] = X_train_cnn
            st.session_state['y_train_cnn'] = y_train_cnn  # Store y_train for comparison

            # Display training loss
            final_loss = history.history['loss'][-1]
            st.subheader("Training Loss")
            st.write(f"Final Training Loss: {final_loss:.6f}")

            # Optional: Plot training loss
            fig_loss, ax_loss = plt.subplots()
            ax_loss.plot(history.history['loss'], label='Training Loss')
            if 'val_loss' in history.history:
                ax_loss.plot(history.history['val_loss'], label='Validation Loss')
            ax_loss.set_title('Loss Curves')
            ax_loss.set_xlabel('Iteration')
            ax_loss.set_ylabel('Loss')
            ax_loss.legend()
            st.pyplot(fig_loss)

        elif model_type == "GAN":
            # Ensure pinn_iterations is defined
            if 'pinn_iterations' not in locals():
                pinn_iterations = st.session_state.get('pinn_iterations', 5000)
            st.write(f"**Iterations**: {pinn_iterations}")  # Moved inside the block
            st.write(f"**Latent Dimension**: {latent_dim}")
            st.write(f"**Generator Units**: {generator_units}")
            st.write(f"**Discriminator Units**: {discriminator_units}")
            st.write(f"**Batch Size**: {batch_size}")

            # Prepare data for GAN
            data_gan = (data - 0.5) * 2  # Scale data to [-1, 1] for tanh activation
            data_gan = data_gan.reshape(data_gan.shape[0], Nx, Ny, 1)

            # Build GAN models
            def build_generator(latent_dim):
                input_layer = Input(shape=(latent_dim,))
                x = Dense(generator_units)(input_layer)
                x = LeakyReLU(0.2)(x)
                x = BatchNormalization()(x)
                x = Dense(Nx * Ny, activation="tanh")(x)
                x = Reshape((Nx, Ny, 1))(x)
                model = Model(inputs=input_layer, outputs=x)
                return model

            def build_discriminator():
                input_layer = Input(shape=(Nx, Ny, 1))
                x = Flatten()(input_layer)
                x = Dense(discriminator_units)(x)
                x = LeakyReLU(0.2)(x)
                x = Dense(1, activation="sigmoid")(x)
                model = Model(inputs=input_layer, outputs=x)
                return model

            # Clear previous models to prevent memory issues
            tf.keras.backend.clear_session()

            # Build and compile the discriminator
            discriminator = build_discriminator()
            discriminator.compile(
                optimizer=Adam(0.0002, 0.5),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )

            # Build the generator
            generator = build_generator(latent_dim)

            # Clone the discriminator for the GAN model
            discriminator_for_gan = clone_model(discriminator)
            discriminator_for_gan.set_weights(discriminator.get_weights())
            for layer in discriminator_for_gan.layers:
                layer.trainable = False

            # Build and compile the GAN model
            gan_input = Input(shape=(latent_dim,))
            gan_output = discriminator_for_gan(generator(gan_input))
            gan = Model(inputs=gan_input, outputs=gan_output)
            gan.compile(
                optimizer=Adam(0.0002, 0.5),
                loss="binary_crossentropy",
            )

            # Train GAN
            st.subheader("Training Progress")
            progress_bar = st.progress(0)
            status_text = st.empty()

            half_batch = int(batch_size / 2)
            num_batches = int(data_gan.shape[0] / half_batch)
            total_steps = pinn_iterations * num_batches
            step = 0

            for epoch in range(pinn_iterations):
                for _ in range(num_batches):
                    step += 1

                    # ---------------------
                    #  Train Discriminator
                    # ---------------------

                    # Sample real images
                    idx = np.random.randint(0, data_gan.shape[0], half_batch)
                    real_data_batch = data_gan[idx]

                    # Generate fake images
                    noise = np.random.normal(0, 1, (half_batch, latent_dim)).astype(np.float32)
                    generated_data_batch = generator.predict(noise)

                    # Combine real and fake data
                    X = np.concatenate([real_data_batch, generated_data_batch])
                    y = np.concatenate([np.ones((half_batch, 1)), np.zeros((half_batch, 1))])

                    # Train the discriminator
                    d_loss = discriminator.train_on_batch(X, y)

                    # -----------------
                    #  Train Generator
                    # -----------------

                    # Generate noise for generator training
                    noise = np.random.normal(0, 1, (batch_size, latent_dim)).astype(np.float32)

                    # Train the generator via the GAN model
                    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

                    # Update training progress every 100 steps
                    if step % 100 == 0 or step == total_steps:
                        status_text.text(
                            f"Iteration {step}/{total_steps}, D Loss: {d_loss[0]:.4f}, G Loss: {g_loss:.4f}"
                        )
                        progress_bar.progress(step / total_steps)

            st.success("GAN Model Trained")

            # Store the trained models in session state
            st.session_state["generator"] = generator
            st.session_state["discriminator"] = discriminator
            st.session_state["model_type"] = model_type
            st.session_state["latent_dim"] = latent_dim

        elif model_type == "PINNs":
            # Ensure pinn_iterations is defined
            if 'pinn_iterations' not in locals():
                pinn_iterations = st.session_state.get('pinn_iterations', 5000)
            st.write(f"**Iterations**: {pinn_iterations}")  # Moved inside the block
            st.write(f"**Learning Rate**: {learning_rate}")

            import os
            os.environ["DDE_BACKEND"] = "tensorflow"
            import deepxde as dde
            from deepxde.backend import tf

            # Prepare data for PINNs
            data_pinn = prepare_data_pinn(data, Lx, Ly, T, beta)

            # Define the PINN model
            def rossby_wave_pde(x, y):
                psi = y
                dpsi_t = dde.grad.jacobian(psi, x, i=0, j=2)
                dpsi_xx = dde.grad.hessian(psi, x, i=0, j=0)
                dpsi_yy = dde.grad.hessian(psi, x, i=1, j=1)
                laplacian = dpsi_xx + dpsi_yy
                return dpsi_t + beta * dde.grad.jacobian(psi, x, i=0, j=0) - laplacian

            geom = dde.geometry.geometry_2d.Rectangle([0, 0], [Lx, Ly])
            timedomain = dde.geometry.TimeDomain(0, T)
            geomtime = dde.geometry.GeometryXTime(geom, timedomain)
            data_pinn = dde.data.TimePDE(
                geomtime,
                rossby_wave_pde,
                [],
                num_domain=2540,
                num_boundary=80,
                num_initial=160,
            )
            net = dde.maps.FNN([3] + [50] * 3 + [1], "tanh", "Glorot normal")
            model_pinn = dde.Model(data_pinn, net)
            model_pinn.compile("adam", lr=learning_rate)

            # Training progress display
            st.subheader("Training Progress")
            training_bar = st.progress(0)
            training_text = st.empty()

            # Custom training loop to update Streamlit progress
            for i in range(1, pinn_iterations + 1):
                model_pinn.train(iterations=1, display_every=1000000)
                if i % 100 == 0 or i == pinn_iterations:
                    loss = model_pinn.losshistory.loss_train[-1]
                    # Extract scalar from NumPy array
                    loss_scalar = loss.item() if isinstance(loss, np.ndarray) else loss
                    training_text.text(f"Iteration {i}/{pinn_iterations}, Loss: {loss_scalar:.6f}")
                    training_bar.progress(i / pinn_iterations)

            st.success("PINN Model Trained")
            st.session_state["model_pinn"] = model_pinn
            st.session_state["model_type"] = model_type

    else:
        st.warning(
            "Please generate the Rossby wave data first by clicking on 'Generate Rossby Wave'."
        )

# Predictions
st.subheader("Wave Prediction")

if st.button("Run Prediction"):
    if "model_type" in st.session_state:
        model_type_saved = st.session_state["model_type"]
        data = st.session_state["data_full"]  # Retrieve the full data array
        data_min = st.session_state["data_min"]
        data_max = st.session_state["data_max"]
        Nx = st.session_state["Nx"]
        Ny = st.session_state["Ny"]
        beta = st.session_state["beta"]
        T = st.session_state["T"]
        Lx = st.session_state["Lx"]
        Ly = st.session_state["Ly"]

        if model_type_saved == "LSTM":
            model = st.session_state["model"]
            sequence_length = st.session_state["sequence_length"]
            X_train_lstm = st.session_state["X_train_lstm"]
            y_train_lstm = st.session_state["y_train_lstm"]
            data_min = st.session_state["data_min"]
            data_max = st.session_state["data_max"]

            input_dim = Nx * Ny

            # Prepare input for prediction
            input_sequence = X_train_lstm[-1].reshape(1, sequence_length, input_dim)
            prediction = model.predict(input_sequence)
            # Reshape prediction
            prediction = prediction.reshape(Nx, Ny)
            # Denormalize prediction
            prediction = prediction * (data_max - data_min) + data_min

            # Get actual wave state
            actual = y_train_lstm[-1].reshape(Nx, Ny)
            actual = actual * (data_max - data_min) + data_min

            # Compute error between prediction and actual
            error = np.mean((prediction - actual) ** 2)
            st.subheader("Prediction Error")
            st.write(f"Mean Squared Error between Predicted and Actual Wave: {error:.6f}")

            # Display images side by side
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("Initial Wave State")
                initial_wave = st.session_state["initial_wave"]
                fig0, ax0 = plt.subplots()
                cax0 = ax0.imshow(
                    initial_wave, cmap="RdBu_r", origin="lower"
                )
                ax0.set_title("Initial Wave State")
                fig0.colorbar(cax0, ax=ax0, label="Streamfunction ψ")
                st.pyplot(fig0)

            with col2:
                st.subheader("Actual Wave State")
                fig1, ax1 = plt.subplots()
                cax1 = ax1.imshow(actual, cmap="RdBu_r", origin="lower")
                ax1.set_title("Actual Wave State")
                fig1.colorbar(cax1, ax=ax1, label="Streamfunction ψ")
                st.pyplot(fig1)

            with col3:
                st.subheader("Predicted Wave State")
                fig2, ax2 = plt.subplots()
                cax2 = ax2.imshow(prediction, cmap="RdBu_r", origin="lower")
                ax2.set_title("Predicted Wave State")
                fig2.colorbar(cax2, ax=ax2, label="Streamfunction ψ")
                st.pyplot(fig2)

            st.write(
                "The **Initial Wave State** is the starting point of the simulation."
            )
            st.write(
                "The **Actual Wave State** is the true state at the next time step."
            )
            st.write(
                "The **Predicted Wave State** is the model's prediction of the next state based on the input data."
            )
            st.write(
                "Comparing the actual and predicted wave states helps evaluate the model's performance."
            )

        elif model_type_saved == "CNN":
            model = st.session_state["model"]
            X_train_cnn = st.session_state["X_train_cnn"]
            y_train_cnn = st.session_state["y_train_cnn"]
            data_min = st.session_state["data_min"]
            data_max = st.session_state["data_max"]

            # Retrieve Nx and Ny after data preparation
            Nx = X_train_cnn.shape[1]
            Ny = X_train_cnn.shape[2]

            # Prepare input for prediction
            input_data = X_train_cnn[-1].reshape(1, Nx, Ny, 1)
            prediction = model.predict(input_data)
            # Reshape prediction to [Nx, Ny]
            prediction = prediction.reshape(Nx, Ny)
            # Denormalize prediction
            prediction = prediction * (data_max - data_min) + data_min

            # Get actual wave state
            actual = y_train_cnn[-1].reshape(Nx, Ny)
            actual = actual * (data_max - data_min) + data_min

            # Compute error between prediction and actual
            error = np.mean((prediction - actual) ** 2)
            st.subheader("Prediction Error")
            st.write(
                f"Mean Squared Error between Predicted and Actual Wave: {error:.6f}"
            )

            # Display images side by side
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("Initial Wave State")
                initial_wave = st.session_state["initial_wave"]
                fig0, ax0 = plt.subplots()
                cax0 = ax0.imshow(
                    initial_wave, cmap="RdBu_r", origin="lower"
                )
                ax0.set_title("Initial Wave State")
                fig0.colorbar(cax0, ax=ax0, label="Streamfunction ψ")
                st.pyplot(fig0)

            with col2:
                st.subheader("Actual Wave State")
                fig1, ax1 = plt.subplots()
                cax1 = ax1.imshow(actual, cmap="RdBu_r", origin="lower")
                ax1.set_title("Actual Wave State")
                fig1.colorbar(cax1, ax=ax1, label="Streamfunction ψ")
                st.pyplot(fig1)

            with col3:
                st.subheader("Predicted Wave State")
                fig2, ax2 = plt.subplots()
                cax2 = ax2.imshow(prediction, cmap="RdBu_r", origin="lower")
                ax2.set_title("Predicted Wave State")
                fig2.colorbar(cax2, ax=ax2, label="Streamfunction ψ")
                st.pyplot(fig2)

            st.write(
                "The **Initial Wave State** is the starting point of the simulation."
            )
            st.write(
                "The **Actual Wave State** is the true state at the next time step."
            )
            st.write(
                "The **Predicted Wave State** is the model's prediction of the next state based on the input data."
            )
            st.write(
                "Comparing the actual and predicted wave states helps evaluate the model's performance."
            )

        elif model_type_saved == "GAN":
            generator = st.session_state["generator"]
            discriminator = st.session_state["discriminator"]
            latent_dim = st.session_state["latent_dim"]
            data_min = st.session_state["data_min"]
            data_max = st.session_state["data_max"]
            Nx = st.session_state["Nx"]
            Ny = st.session_state["Ny"]
            data = st.session_state["data_full"]

            # Generate a new wave
            noise = np.random.normal(0, 1, (1, latent_dim)).astype(np.float32)
            generated_wave = generator.predict(noise)
            generated_wave = generated_wave.reshape(Nx, Ny)
            # Denormalize
            generated_wave = (generated_wave / 2) + 0.5  # Scale back to [0,1]
            generated_wave = generated_wave * (data_max - data_min) + data_min

            # Select an actual wave state for comparison
            actual_wave = data[-1].reshape(Nx, Ny)
            actual_wave = actual_wave * (data_max - data_min) + data_min

            # Display images side by side
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("Initial Wave State")
                initial_wave = st.session_state["initial_wave"]
                fig0, ax0 = plt.subplots()
                cax0 = ax0.imshow(
                    initial_wave, cmap="RdBu_r", origin="lower"
                )
                ax0.set_title("Initial Wave State")
                fig0.colorbar(cax0, ax=ax0, label="Streamfunction ψ")
                st.pyplot(fig0)

            with col2:
                st.subheader("Actual Wave State")
                fig1, ax1 = plt.subplots()
                cax1 = ax1.imshow(actual_wave, cmap="RdBu_r", origin="lower")
                ax1.set_title("Actual Wave State")
                fig1.colorbar(cax1, ax=ax1, label="Streamfunction ψ")
                st.pyplot(fig1)

            with col3:
                st.subheader("Generated Wave (GAN)")
                fig2, ax2 = plt.subplots()
                cax2 = ax2.imshow(
                    generated_wave, cmap="RdBu_r", origin="lower"
                )
                ax2.set_title("Generated Wave (GAN)")
                fig2.colorbar(cax2, ax=ax2, label="Streamfunction ψ")
                st.pyplot(fig2)

            st.write(
                "Comparing the generated wave with an actual wave from the dataset."
            )

        elif model_type_saved == "PINNs":
            model_pinn = st.session_state["model_pinn"]
            data_min = st.session_state["data_min"]
            data_max = st.session_state["data_max"]
            Nx = st.session_state["Nx"]
            Ny = st.session_state["Ny"]
            T = st.session_state["T"]
            Lx = st.session_state["Lx"]
            Ly = st.session_state["Ly"]
            data = st.session_state["data_full"]

            # Generate prediction at specific time
            x = np.linspace(0, Lx, Nx)
            y = np.linspace(0, Ly, Ny)
            X_mesh, Y_mesh = np.meshgrid(x, y)
            X_flat = np.column_stack(
                (X_mesh.ravel(), Y_mesh.ravel(), np.full(Nx * Ny, T))
            )
            y_pred = model_pinn.predict(X_flat)
            wave_state = y_pred.reshape(Nx, Ny)  # Adjust shape if needed

            # Denormalize
            wave_state = wave_state * (data_max - data_min) + data_min

            # Get actual wave state
            actual_state = data[-1].reshape(Nx, Ny)
            actual_state = actual_state * (data_max - data_min) + data_min

            # Display images side by side
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("Initial Wave State")
                initial_wave = st.session_state["initial_wave"]
                fig0, ax0 = plt.subplots()
                cax0 = ax0.imshow(
                    initial_wave, cmap="RdBu_r", origin="lower"
                )
                ax0.set_title("Initial Wave State")
                fig0.colorbar(cax0, ax=ax0, label="Streamfunction ψ")
                st.pyplot(fig0)

            with col2:
                st.subheader("Actual Wave State")
                fig1, ax1 = plt.subplots()
                cax1 = ax1.imshow(actual_state, cmap="RdBu_r", origin="lower")
                ax1.set_title("Actual Wave State")
                fig1.colorbar(cax1, ax=ax1, label="Streamfunction ψ")
                st.pyplot(fig1)

            with col3:
                st.subheader("Predicted Wave (PINN)")
                fig2, ax2 = plt.subplots()
                cax2 = ax2.imshow(wave_state, cmap="RdBu_r", origin="lower")
                ax2.set_title("Predicted Wave (PINN)")
                fig2.colorbar(cax2, ax=ax2, label="Streamfunction ψ")
                st.pyplot(fig2)

            st.write(
                "Comparing the predicted wave from PINN with the initial and actual wave states."
            )

    else:
        st.warning("Please train the model first.")

