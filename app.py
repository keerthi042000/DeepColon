import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from custom_layers.scale_layer import Scale  # your custom layer
import os
import warnings
# -----------------------------
# Suppress warnings and TF logs
# -----------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress TF messages
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="DeepColon",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <div style="text-align:center; background-color:#00695C; padding:20px; border-radius:12px">
        <h1 style="color:white;">DeepColon</h1>
        <p style="color:white; font-size:18px;">Histopathological Colon Tissue Classification</p>
        <p style="color:white; font-size:18px;">Upload a histopathological image of colon tissue to detect if it is cancerous or benign.</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

@tf.keras.utils.register_keras_serializable()
class Autoencoder(tf.keras.Model):
    def __init__(self, input_shape):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same')
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.UpSampling2D((2, 2)),
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.UpSampling2D((2, 2)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.UpSampling2D((2, 2)),
            tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

    @classmethod
    def from_config(cls, config):
        autoencoder = cls(input_shape=(224, 224, 3))
        return autoencoder


# Load models (once, when the app starts)
@st.cache_resource
def load_models():
    autoencoder = tf.keras.models.load_model(
        "model_output_autoencoder_working100/model.keras",
        custom_objects={"Autoencoder": Autoencoder}
    )
    densenet = tf.keras.models.load_model(
        "model_output_densenet_working100/model.keras",
        compile=False,
        custom_objects={"Scale": Scale}
    )
    return autoencoder, densenet

autoencoder, densenet = load_models()

# st.title("Colon Tissue Classification")
# -----------------------------
# Streamlit UI
# -----------------------------
# st.set_page_config(
#     page_title="Colon Cancer Classifier",
#     layout="centered",
#     initial_sidebar_state="expanded"
# )

# st.set_page_config(
#     page_title="DeepColon",
#     page_icon="🩺",
#     layout="centered",
#     initial_sidebar_state="expanded"
# )

# st.set_page_config(page_title="Colon Cancer Classifier", page_icon="🩺", layout="centered")
# st.title("Colon Tissue Cancer Detection (Histopathology)")
# st.markdown("Upload a histopathological image of colon tissue to detect if it is cancerous or benign.")
with st.sidebar:
    st.header("About")
    st.markdown("""
    - **Model:** Autoencoder + DenseNet
    - **Dataset:** Histopathological colon tissue images
    - **Purpose:** Predict colon adenocarcinoma vs benign tissue
    """)

# Initialize a session state variable to track uploaded file
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

# Upload image
uploaded_file = st.file_uploader("Upload a colon tissue image", type=["jpg", "jpeg", "png"])
# uploaded_file = st.file_uploader("Upload Histopathology Image (JPEG/PNG)", type=["jpg", "jpeg", "png"])

if st.button("Clear"):
    # uploaded_file = None
    # st.experimental_rerun()
    st.session_state.uploaded_file = None

if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file

if st.session_state.uploaded_file:
    uploaded_file = st.session_state.uploaded_file
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0


    with st.spinner("Classifying... 🔬"):
        _, activation_map = autoencoder.predict(img_array)
        pred = densenet.predict(activation_map)
        pred_class = np.argmax(pred, axis=1)[0]
        pred_prob = np.round(pred[0] * 100, 2)
    
    # Get activation map from autoencoder
    # _, activation_map = autoencoder.predict(img_array)

    # # Predict using DenseNet
    # pred = densenet.predict(activation_map)
    # pred_class = np.argmax(pred, axis=1)[0]

    # pred_prob = np.max(pred)

    # Map to human-readable labels
    # labels = ["Colon Benign Tissue (Non-Cancerous)", "Colon Adenocarcinoma (Cancerous)"]
    # st.image(img, caption="Uploaded Image", use_column_width=True)
    # st.markdown(f"**Prediction:** {labels[pred_class]}")
    # st.markdown(f"**Confidence:** {pred_prob*100:.2f}%")
    # Map index to label
    class_labels = ["Colon Adenocarcinoma", "Colon Benign Tissue"]
    st.success(f"**Predicted Class:** {class_labels[pred_class]}")
    st.info(f"**Confidence:** {pred_prob[pred_class]}%")



    if st.checkbox("Show all class probabilities"):
        prob_dict = {class_labels[i]: f"{pred_prob[i]}%" for i in range(len(class_labels))}
        st.table(prob_dict)

    # Optional: activation map visualization
    if st.checkbox("Show activation map (encoded features)"):
        activation_img = activation_map[0]
        # Convert first channel to grayscale image
        act_map = np.mean(activation_img, axis=-1)
        act_map = (act_map - act_map.min()) / (act_map.max() - act_map.min())  # normalize
        st.image(act_map, caption="Activation Map (Grayscale)", use_column_width=True)

    # Map to human-readable labels
    # if pred_class == 0:
    #     st.success("Patient has **Colon Adenocarcinoma** (Cancer).")
    # else:
    #     st.info("Patient has **Colon Benign Tissue** (No Cancer).")

        # Display results
    # pred_prob = np.max(pred)
    # if pred_class == 0:
    #     st.success(f"Predicted: **Colon Adenocarcinoma** (Cancer) with probability {pred_prob:.2f}")
    # else:
    #     st.info(f"Predicted: **Colon Benign Tissue** (No Cancer) with probability {pred_prob:.2f}")

    # st.markdown("---")
    # st.markdown("💡 **Note:** This tool is for educational purposes and should not replace medical advice.")
