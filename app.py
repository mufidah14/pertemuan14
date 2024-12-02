import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pickle
import os

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    data = pd.read_csv('diabetes.csv')
    return data

# Fungsi untuk memuat model
@st.cache_resource
def load_model():
    with open('model/diabetes_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Sidebar Menu
menu = st.sidebar.selectbox(
    "Pilih Menu",
    options=["Home", "Tampilkan Dataset", "Visualisasi Data", "Prediksi Diabetes"]
)

# Menu: Home
if menu == "Home":
    st.title("Prediksi Diabetes Menggunakan Machine Learning")
    st.image(
        "kesehatan diabetes.jpg", 
        caption="Ilustrasi Prediksi Diabetes",
        use_container_width=True
    )

    st.markdown("""
    Selamat datang di aplikasi **Prediksi Diabetes**! Aplikasi ini bertujuan untuk membantu Anda 
    memahami risiko diabetes berdasarkan data kesehatan Anda. Anda dapat melihat dataset yang digunakan, 
    memvisualisasikan data, dan melakukan prediksi risiko diabetes secara interaktif.
    """)

# Menu: Tampilkan Dataset
elif menu == "Tampilkan Dataset":
    st.title("Dataset Diabetes")
    data = load_data()
    st.write("Berikut adalah dataset yang digunakan dalam aplikasi ini:")
    st.dataframe(data)
    st.write(f"**Total Data:** {data.shape[0]} baris dan {data.shape[1]} kolom.")
    
    # Menampilkan statistik deskriptif
    st.subheader("Statistik Deskriptif")
    st.write(data.describe())

# Menu: Visualisasi Data
elif menu == "Visualisasi Data":
    st.title("Visualisasi Data")
    data = load_data()

    # Grafik Distribusi Glukosa
    st.subheader("Distribusi Kadar Glukosa")
    fig, ax = plt.subplots()
    sns.histplot(data['Glucose'], kde=True, color="blue", ax=ax)
    ax.set_xlabel("Kadar Glukosa")
    ax.set_ylabel("Frekuensi")
    st.pyplot(fig)

    # Grafik Perbandingan Outcome
    st.subheader("Perbandingan Outcome (Negatif vs Positif Diabetes)")
    fig, ax = plt.subplots()
    sns.countplot(x='Outcome', data=data, palette="Set2", ax=ax)
    ax.set_xticklabels(["Negatif Diabetes", "Positif Diabetes"])
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Jumlah")
    st.pyplot(fig)

    # Heatmap Korelasi Antar Fitur
    st.subheader("Heatmap Korelasi Antar Fitur")
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = data.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # WordCloud Fitur
    st.subheader("WordCloud Fitur")
    wordcloud_data = ' '.join(data.columns)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(wordcloud_data)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

# Menu: Prediksi Diabetes
elif menu == "Prediksi Diabetes":
    st.title("Prediksi Diabetes")
    st.write("Masukkan data kesehatan Anda di bawah ini untuk memprediksi kemungkinan diabetes.")

    # Input pengguna
    glucose = st.number_input("Kadar Glukosa", min_value=0, max_value=200, value=85)
    blood_pressure = st.number_input("Tekanan Darah (mm Hg)", min_value=0, max_value=150, value=70)
    skin_thickness = st.number_input("Ketebalan Kulit (mm)", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin (IU/mL)", min_value=0, max_value=900, value=79)
    bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input("Usia", min_value=0, max_value=120, value=33)

    # Prediksi
    if st.button("Prediksi"):
        try:
            # Membuat DataFrame dari input
            input_data = pd.DataFrame([[glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]], 
                                      columns=['Glucose', 'BloodPressure', 'SkinThickness', 
                                               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
            
            # Load model
            model = load_model()
            
            # Prediksi
            prediction = model.predict(input_data)[0]
            result = "Positif Diabetes" if prediction == 1 else "Negatif Diabetes"
            
            st.success(f"Hasil prediksi: **{result}**")
        except FileNotFoundError:
            st.error("Model tidak ditemukan! Pastikan file 'diabetes_model.pkl' ada di folder 'model'.")
        except Exception as e:
            st.error(f"Terjadi kesalahan dalam prediksi: {e}")
