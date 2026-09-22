import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Sayfa Konfigürasyonu
st.set_page_config(
    page_title="Kalp Yetmezliği KDKS - Giriş",
    page_icon="🩺",
    layout="wide"
)

# ---------------------------------------------------------
# 1. GÜVENLİK VE GİRİŞ KONTROLÜ (SESSION STATE)
# ---------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

def check_login(username, password):
    # İstersen bu kullanıcı adı ve şifreyi değiştirebilirsin
    VALID_USERNAME = "doktor"
    VALID_PASSWORD = "123"
    
    if username == VALID_USERNAME and password == VALID_PASSWORD:
        st.session_state["authenticated"] = True
        st.rerun()
    else:
        st.error("❌ Hatalı kullanıcı adı veya şifre!")

def logout():
    st.session_state["authenticated"] = False
    st.rerun()

# ---------------------------------------------------------
# 2. GİRİŞ EKRANI (LOGGED OUT)
# ---------------------------------------------------------
if not st.session_state["authenticated"]:
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<h2 style='text-align: center;'>🩺 Kardiyoloji Karar Destek Sistemi</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: gray;'>Klinik Kullanım İçin Güvenli Giriş Paneli</p>", unsafe_allow_html=True)
        st.divider()
        
        with st.form("login_form"):
            username_input = st.text_input("Kullanıcı Adı", placeholder="Örn: doktor")
            password_input = st.text_input("Şifre", type="password", placeholder="***")
            submit_button = st.form_submit_button("Giriş Yap", use_container_width=True)
            
            if submit_button:
                check_login(username_input, password_input)
                
        st.info("💡 **Demo Giriş Bilgileri:**\n- **Kullanıcı Adı:** `doktor` \n- **Şifre:** `123`")
    st.stop()  # Giriş yapılmadıysa uygulamanın geri kalanını çalıştırma!

# ---------------------------------------------------------
# 3. ESAS UYGULAMA EKRANI (LOGGED IN)
# ---------------------------------------------------------

# Model ve Scaler Yükleme
@st.cache_resource
def load_artifacts():
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    data = np.load('processed_data.npz', allow_pickle=True)
    feature_names = data['feature_names']
    return model, scaler, feature_names

model, scaler, feature_names = load_artifacts()

# Header ve Çıkış Butonu
st.sidebar.markdown("### 👤 Oturum Bilgisi")
st.sidebar.success("Giriş Yapıldı: **Dr. Ahmet**")
if st.sidebar.button("🚪 Güvenli Çıkış Yap", use_container_width=True):
    logout()

st.sidebar.divider()

# Yan Panel / Form Girişleri
st.sidebar.header("📋 Hasta Klinik Verileri")

age = st.sidebar.slider("Yaş", 18, 100, 50)
sex = st.sidebar.selectbox("Cinsiyet", ["Erkek (M)", "Kadın (F)"])
chest_pain = st.sidebar.selectbox("Göğüs Ağrısı Tipi (ChestPainType)", ["ATA", "NAP", "ASY", "TA"])
resting_bp = st.sidebar.number_input("Dinlenim Kan Basıncı (RestingBP - mm Hg)", 80, 200, 120)
cholesterol = st.sidebar.number_input("Kolesterol (Cholesterol - mm/dl)", 0, 600, 200)
fasting_bs = st.sidebar.selectbox("Açlık Kan Şekeri > 120 mg/dl (FastingBS)", [0, 1], format_func=lambda x: "Evet (1)" if x == 1 else "Hayır (0)")
resting_ecg = st.sidebar.selectbox("Dinlenim EKG (RestingECG)", ["Normal", "ST", "LVH"])
max_hr = st.sidebar.slider("Maksimum Nabız (MaxHR)", 60, 220, 150)
exercise_angina = st.sidebar.selectbox("Egzersiz Anjini var mı? (ExerciseAngina)", ["N", "Y"], format_func=lambda x: "Evet (Y)" if x == "Y" else "Hayır (N)")
oldpeak = st.sidebar.number_input("Oldpeak (ST Depresyonu)", 0.0, 10.0, 1.0, step=0.1)
st_slope = st.sidebar.selectbox("ST Eğim Tipi (ST_Slope)", ["Up", "Flat", "Down"])

# Kullanıcı verilerini hazırlama
input_dict = {
    'Age': age,
    'RestingBP': resting_bp,
    'Cholesterol': cholesterol,
    'FastingBS': fasting_bs,
    'MaxHR': max_hr,
    'Oldpeak': oldpeak,
    'Sex_M': 1 if sex == "Erkek (M)" else 0,
    'ChestPainType_ATA': 1 if chest_pain == "ATA" else 0,
    'ChestPainType_NAP': 1 if chest_pain == "NAP" else 0,
    'ChestPainType_TA': 1 if chest_pain == "TA" else 0,
    'RestingECG_Normal': 1 if resting_ecg == "Normal" else 0,
    'RestingECG_ST': 1 if resting_ecg == "ST" else 0,
    'ExerciseAngina_Y': 1 if exercise_angina == "Y" else 0,
    'ST_Slope_Flat': 1 if st_slope == "Flat" else 0,
    'ST_Slope_Up': 1 if st_slope == "Up" else 0,
}

input_df = pd.DataFrame([input_dict])
for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[feature_names]

# Ana Ekran
st.title("🩺 Klinik Karar Destek Sistemi: Kalp Yetmezliği Risk Tahmini")
st.markdown("Bu panel üzerinden hastaların klinik parametrelerini girerek anlık yapay zekâ risk analizi alabilirsiniz.")
st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🔍 Hasta Özet Tablosu")
    st.dataframe(pd.DataFrame({
        "Parametre": ["Yaş", "Cinsiyet", "Göğüs Ağrısı", "Kan Basıncı", "Kolesterol", "Maks. Nabız", "Oldpeak"],
        "Değer": [age, sex, chest_pain, resting_bp, cholesterol, max_hr, oldpeak]
    }), use_container_width=True)

with col2:
    st.subheader("📊 Tahmin & Risk Analizi")
    
    if st.button("🚀 Risk Oranını Hesapla", use_container_width=True):
        scaled_input = scaler.transform(input_df)
        risk_probability = model.predict_proba(scaled_input)[0][1] * 100
        prediction = model.predict(scaled_input)[0]
        
        st.markdown("---")
        if prediction == 1:
            st.error("⚠️ **YÜKSEK KALP HASTALIĞI RİSKİ!**")
            st.metric(label="Tahmini Risk Seviyesi", value=f"%{risk_probability:.1f}")
            st.warning("Hastanın ileri kardiyolojik değerlendirmeye sevk edilmesi önerilir.")
        else:
            st.success("✅ **DÜŞÜK KALP HASTALIĞI RİSKİ**")
            st.metric(label="Tahmini Risk Seviyesi", value=f"%{risk_probability:.1f}")
            st.info("Klinik bulgular normal sınırlar içerisinde görünmektedir.")

st.divider()

st.subheader("💡 Model Karar Faktörleri (Özellik Önem Seviyeleri)")
if hasattr(model, 'feature_importances_'):
    fig, ax = plt.subplots(figsize=(8, 4))
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)
    importances.tail(8).plot(kind='barh', ax=ax, color='#1f77b4')
    ax.set_title("Model İçin En Belirleyici Top 8 Klinik Parametre")
    ax.set_xlabel("Göreli Önem Derecesi")
    st.pyplot(fig)