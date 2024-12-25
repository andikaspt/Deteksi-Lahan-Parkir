from ultralytics import YOLO
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os

# Configure Streamlit page
st.set_page_config(
    page_title="🚗 Dashboard Deteksi Lahan Parkir Kosong",
    page_icon="🚗",
    layout="wide",
)

# Tambahkan CSS kustom
def add_custom_css():
    custom_css = """
    <style>
        /* Warna latar belakang */
        body {
            background-color: #f0f8ff;
        }

        /* Warna latar belakang sidebar */
        [data-testid="stSidebar"] {
            background-color: #add8e6;
        }

        /* Header dan subheader */
        h1, h2, h3, h4, h5, h6 {
            color: #2e86c1;
        }

        /* Tombol */
        button {
            background-color: #1abc9c;
            color: white;
            border-radius: 5px;
            padding: 10px;
        }
        button:hover {
            background-color: #16a085;
        }

        /* Teks di sidebar */
        [data-testid="stSidebar"] .css-1q8dd3e {
            color: #ffffff;
        }

        /* Slider */
        .stSlider .css-10trblm {
            background-color: #1abc9c;
        }

        /* Placeholder untuk gambar dan hasil */
        .css-1hs6j8u {
            border: 1px solid #1abc9c;
            padding: 10px;
            border-radius: 5px;
        }

        /* Footer */
        footer {
            background-color: #add8e6;
            padding: 10px;
            color: #2e86c1;
            border-top: 2px solid #2e86c1;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

# Tambahkan CSS
add_custom_css()

# Load YOLO model
model_path = "best.pt"  # Path to saved model
model = YOLO(model_path)

# Page title and description
st.title("🚗 Dashboard Deteksi Lahan Parkir Kosong")
st.markdown(
    """
    Selamat datang di dashboard deteksi lahan parkir kosong menggunakan YOLO. Anda dapat:
    - Mengunggah gambar untuk deteksi.
    - Menggunakan kamera untuk deteksi secara real-time.
    - Mengunggah video untuk analisis frame-by-frame.
    """
)

# Sidebar menu
st.sidebar.title("Menu")
option = st.sidebar.radio("Pilih Mode Deteksi:", ["Unggah Gambar", "Gunakan Kamera", "Unggah Video"])

# Sidebar settings
st.sidebar.header("Pengaturan")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# Function to process video or webcam feed
def process_video(video_source):
    cap = cv2.VideoCapture(video_source)
    frame_placeholder = st.empty()  # Placeholder untuk menampilkan frame
    slot_info_placeholder = st.empty()  # Placeholder untuk menampilkan info slot parkir

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Tidak dapat membaca frame dari kamera atau video.")
            break

        # YOLO prediction
        results = model.predict(source=frame, imgsz=640, conf=confidence_threshold)
        annotated_frame = results[0].plot()

        # Count total parking slots and empty slots
        labels = results[0].boxes.cls.tolist()  # Mendapatkan daftar kelas deteksi
        names = results[0].names               # Nama kelas dari model

        total_slots = len(labels)  # Jumlah total slot parkir
        empty_slots = sum(1 for label in labels if names[int(label)] == "free")  # Jumlah slot kosong

        # Update placeholders
        frame_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)
        slot_info_placeholder.markdown(
            f"""**Jumlah total lahan parkir:** **{total_slots}**  
            **Jumlah lahan parkir kosong:** **{empty_slots}**"""
        )

    cap.release()

# Process based on selected option
if option == "Unggah Gambar":
    uploaded_file = st.sidebar.file_uploader("Pilih gambar Anda:", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.subheader("📤 Gambar yang Diupload")
        image = Image.open(uploaded_file)

        # Convert to NumPy array
        image_np = np.array(image)

        # Predict with YOLO
        with st.spinner("🚀 Memproses gambar..."):
            results = model.predict(source=image_np, imgsz=640, conf=confidence_threshold)

        # Display original and detection results side-by-side
        st.subheader("📊 Hasil Deteksi")
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Gambar Asli", use_column_width=True)

        with col2:
            st.image(results[0].plot(), caption="Hasil Deteksi", use_column_width=True)

        # Count total parking slots and empty slots
        labels = results[0].boxes.cls.tolist()  # Mendapatkan daftar kelas deteksi
        names = results[0].names               # Nama kelas dari model

        total_slots = len(labels)  # Jumlah total slot parkir
        empty_slots = sum(1 for label in labels if names[int(label)] == "free")  # Jumlah slot kosong

        # Show detection counts
        st.success(f"Jumlah total lahan parkir: **{total_slots}**")
        st.success(f"Jumlah lahan parkir kosong: **{empty_slots}**")
    else:
        st.info("Silakan unggah gambar melalui sidebar untuk memulai deteksi!")

elif option == "Gunakan Kamera":
    st.subheader("📸 Deteksi Menggunakan Kamera")
    st.write("Klik tombol di bawah untuk memulai deteksi menggunakan kamera.")
    camera_index = st.sidebar.selectbox("Pilih Kamera", options=[0, 1], format_func=lambda x: f"Kamera {x}")

    if st.button("Mulai Kamera"):
        process_video(camera_index)

elif option == "Unggah Video":
    uploaded_video = st.sidebar.file_uploader("Unggah video Anda:", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        st.subheader("🎥 Video yang Diupload")
        st.video(uploaded_video)

        # Save video temporarily
        video_path = f"temp_{uploaded_video.name}"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())

        # Process video
        st.write("Proses deteksi sedang berlangsung...")
        process_video(video_path)

        # Clean up temporary file
        os.remove(video_path)
    else:
        st.info("Silakan unggah video melalui sidebar untuk memulai deteksi!")

# Footer
st.markdown("---")
st.markdown(
    """
    **Catatan**:
    - Model ini menggunakan YOLO untuk deteksi objek. Jika hasil tidak sesuai harapan, coba gambar atau video dengan resolusi lebih tinggi.
    """
)
