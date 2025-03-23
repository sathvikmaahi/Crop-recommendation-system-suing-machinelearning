import streamlit as st
import numpy as np
import joblib
from transformers import pipeline
import matplotlib.pyplot as plt

# Load ML model
@st.cache_resource
def load_model():
    return joblib.load("/Users/sathviksanka/Desktop/crop_project/best_model.pkl")

model = load_model()

# Crop explanation dictionary
CROP_REASONS = {
    "rice": "Rice grows well in areas with high humidity, heavy rainfall (more than 150 mm), and slightly acidic to neutral soil (pH 5.5 to 7). It requires warm temperatures (around 20-30°C) and nitrogen-rich soil.",
    "maize": "Maize prefers warm temperatures (21-27°C), moderate rainfall, and slightly acidic to neutral soil (pH 5.5 to 7.5). It grows well in nitrogen-rich, well-drained soils with good sunlight exposure.",
    "chickpea": "Chickpeas require cooler, dry climates and grow best in loamy, well-drained soils. Ideal temperature is 18-30°C, and they tolerate moderate rainfall. Optimal pH is around 6 to 7.5.",
    "kidneybeans": "Kidney beans grow well in warm climates with moderate rainfall. They require well-drained, fertile soil with a pH between 6 and 7.5, and prefer moderate nitrogen content.",
    "pigeonpeas": "Pigeon peas require a warm climate (25-35°C), moderate rainfall, and well-drained soil. Ideal soil pH is 5.5 to 7. They are drought-resistant and suited for semi-arid conditions.",
    "mothbeans": "Moth beans are drought-resistant and grow in dry, arid regions with high temperatures and low rainfall. They prefer sandy or loamy soil with a pH range of 6.2 to 7.5.",
    "mungbean": "Mung beans prefer warm temperatures (25-35°C), moderate humidity, and loamy soil with good drainage. They grow best in soil with pH 6.2 to 7.2 and require moderate rainfall.",
    "blackgram": "Black gram grows in tropical climates with moderate rainfall and warm temperatures (25-30°C). It requires fertile, loamy soil with a pH of 6.0 to 7.5.",
    "lentil": "Lentils require cool weather during early stages and warm weather during maturation. Ideal pH is 6.0 to 8.0. They prefer loamy soils with moderate nitrogen content and low humidity.",
    "pomegranate": "Pomegranates thrive in hot, dry climates with less humidity. They need well-drained sandy loam soil with a pH of 5.5 to 7.2 and moderate water supply.",
    "banana": "Bananas need a humid, tropical climate with high temperatures (26-30°C) and high rainfall. They prefer deep, well-drained loamy soil with a pH of 6.0 to 7.5.",
    "mango": "Mangoes prefer tropical to subtropical climates, moderate rainfall, and well-drained soil. Ideal temperature is 24-30°C with pH ranging from 5.5 to 7.5.",
    "grapes": "Grapes require a warm, dry climate and loamy soil with good drainage. Optimal temperature is 20-30°C, and the ideal soil pH is 5.5 to 6.5.",
    "watermelon": "Watermelons thrive in hot climates with temperatures between 25-35°C and require sandy loam soil. They need low to moderate rainfall and soil pH of 6.0 to 7.5.",
    "muskmelon": "Muskmelons require warm temperatures (25-30°C), well-drained sandy loam soil, and low to moderate rainfall. Ideal pH is 6.0 to 7.0.",
    "apple": "Apples grow best in cool climates with cold winters and mild summers. Ideal pH is 6.0 to 7.0, and they need well-drained loamy soils and moderate humidity.",
    "orange": "Oranges grow well in subtropical climates with moderate humidity. Ideal temperature is 20-30°C. They require sandy loam soil and a pH range of 5.5 to 7.0.",
    "papaya": "Papayas prefer warm temperatures (25-35°C), high humidity, and well-drained soil. The ideal pH is 6.0 to 6.5 and they need moderate to high rainfall.",
    "coconut": "Coconuts grow well in coastal tropical climates with high humidity and rainfall. They need sandy, well-drained soil and pH between 5.5 and 7.0.",
    "cotton": "Cotton thrives in warm climates with low to moderate rainfall and plenty of sunshine. It requires well-drained, loamy soil with pH between 5.8 to 8.0.",
    "jute": "Jute grows in hot, humid climates with high rainfall and requires loamy alluvial soil. Ideal temperature is 24-37°C, and pH should be 6.0 to 7.5.",
    "coffee": "Coffee requires a tropical climate with moderate rainfall and temperature between 15-28°C. It grows in fertile, well-drained soil with pH 6.0 to 6.5."
}

# Load text2text model
generator = pipeline("text2text-generation", model="google/flan-t5-small")

def explain_prediction(crop, features):
    context = CROP_REASONS.get(crop.lower(), "No specific information available.")
    prompt = (
        f"The recommended crop is {crop}. "
        f"The field has Nitrogen={features[0]}, Phosphorus={features[1]}, Potassium={features[2]}, "
        f"Temperature={features[3]}°C, Humidity={features[4]}%, pH={features[5]}, Rainfall={features[6]} mm. "
        f"Given this data, and knowing that {context} Explain in simple terms why this crop fits these conditions."
    )
    output = generator(prompt, max_length=120)
    return output[0]["generated_text"]

# Streamlit UI
st.title("🌾 Smart Crop Recommendation System")
st.markdown("### Provide your field's parameters below:")

N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50)
P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=50)
K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=50)
temperature = st.number_input("Temperature (°C)", value=25.0)
humidity = st.number_input("Humidity (%)", value=65.0)
ph = st.number_input("Soil pH", value=6.5)
rainfall = st.number_input("Rainfall (mm)", value=100.0)

if st.button("🌿 Recommend Crop"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)[0]
    st.success(f"✅ **Recommended Crop:** {prediction.upper()}")

    # Confidence score
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_data)[0]
        confidence = round(np.max(probs) * 100, 2)
        st.info(f"🌡️ Suitability Confidence: **{confidence}%**")

    # Radar chart visualization
    labels = ["N", "P", "K", "Temp", "Humidity", "pH", "Rainfall"]
    values = [N, P, K, temperature, humidity, ph, rainfall]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title("🌿 Field Condition Overview")
    st.pyplot(fig)

    # LLM Explanation
    with st.spinner("🔍 Generating explanation..."):
        explanation = explain_prediction(prediction, [N, P, K, temperature, humidity, ph, rainfall])
        st.markdown("### 🧠 Why this crop?")
        st.info(explanation)
