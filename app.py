import openai
import streamlit as st
from transformers import pipeline, MarianMTModel, MarianTokenizer
import huggingface_hub
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Keys Configuration
HUGGING_FACE_TOKEN = "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"
OPENAI_API_KEY = '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'

# Set API keys
openai.api_key = OPENAI_API_KEY
huggingface_hub.login(token=HUGGING_FACE_TOKEN)

# Load translation models
@st.cache_resource
def load_translation_models():
    """Load translation models with caching"""
    try:
        return {
            'en_ur_model': MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ur"),
            'en_ur_tokenizer': MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ur"),
            'ur_en_model': MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ur-en"),
            'ur_en_tokenizer': MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ur-en")
        }
    except Exception as e:
        logger.error(f"Error loading translation models: {str(e)}")
        raise

# Load symptom analysis models
@st.cache_resource
def load_symptom_analysis_model():
    """Load symptom analysis model with caching"""
    try:
        models = [
            "yikuan8/ClinicalBERT",
            "emilyalsentzer/Bio_ClinicalBERT",
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        ]

        for model_name in models:
            try:
                model = pipeline("text-classification", model=model_name)
                logger.info(f"Successfully loaded model: {model_name}")
                return model
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {str(e)}")
                continue

        raise Exception("Failed to load any symptom analysis model")
    except Exception as e:
        logger.error(f"Error loading symptom analysis model: {str(e)}")
        raise

class HealthAdviceSystem:
    def __init__(self):
        self.english_to_urdu_model = None
        self.english_to_urdu_tokenizer = None
        self.urdu_to_english_model = None
        self.urdu_to_english_tokenizer = None
        self.symptom_analysis = None

    def initialize_translation_models(self):
        """Initialize translation models with error handling"""
        try:
            models = load_translation_models()
            self.english_to_urdu_model = models['en_ur_model']
            self.english_to_urdu_tokenizer = models['en_ur_tokenizer']
            self.urdu_to_english_model = models['ur_en_model']
            self.urdu_to_english_tokenizer = models['ur_en_tokenizer']
            logger.info("Translation models loaded successfully")
        except Exception as e:
            st.error("Failed to load translation models. Please try again later.")
            raise

    def initialize_symptom_analysis(self):
        """Initialize symptom analysis"""
        try:
            self.symptom_analysis = load_symptom_analysis_model()
            logger.info("Symptom analysis model loaded successfully")
        except Exception as e:
            logger.error(f"Error initializing symptom analysis: {str(e)}")
            st.warning("Symptom analysis functionality may be limited. Basic advice will still work.")
            self.symptom_analysis = None

    def analyze_symptoms(self, symptoms: str):
        """Analyze symptoms with error handling"""
        if not self.symptom_analysis:
            return None

        try:
            # Split symptoms by comma and analyze each
            symptom_list = [s.strip() for s in symptoms.split(',')]
            analyses = []

            for symptom in symptom_list:
                if symptom:
                    analysis = self.symptom_analysis(symptom)
                    analyses.append({
                        'symptom': symptom,
                        'analysis': analysis
                    })

            return analyses
        except Exception as e:
            logger.error(f"Error analyzing symptoms: {str(e)}")
            return None

    def get_gpt_advice(self, age, weight, parental_history, personal_history, symptoms, test_results):
        """Get medical advice from GPT-3.5-turbo with error handling"""
        try:
            messages = [
                {"role": "system", "content": "You are a helpful medical assistant providing health advice."},
                {"role": "user", "content": f"""Provide personalized health advice based on the following information:
                Age: {age}
                Weight: {weight}kg
                Parental Medical History: {parental_history}
                Personal Medical History: {personal_history}
                Symptoms: {symptoms}
                Test Results: {test_results}

                Please provide:
                1. Possible diagnosis (with disclaimer)
                2. Recommended diagnostic tests
                3. General health recommendations
                4. Diet and lifestyle suggestions
                5. When to seek immediate medical attention
                6. Explanation of medical terms in simple language"""}
            ]

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            return response.choices[0].message['content'].strip()
        except Exception as e:
            logger.error(f"Error getting GPT advice: {str(e)}")
            raise

    def translate_to_urdu(self, text):
        """Translate text to Urdu with error handling"""
        try:
            translation = self.english_to_urdu_model.generate(
                **self.english_to_urdu_tokenizer.prepare_seq2seq_batch([text], return_tensors="pt")
            )
            return self.english_to_urdu_tokenizer.decode(translation[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Translation to Urdu failed: {str(e)}")
            raise

    def translate_to_english(self, text):
        """Translate text to English with error handling"""
        try:
            translation = self.urdu_to_english_model.generate(
                **self.urdu_to_english_tokenizer.prepare_seq2seq_batch([text], return_tensors="pt")
            )
            return self.urdu_to_english_tokenizer.decode(translation[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Translation to English failed: {str(e)}")
            raise

def main():
    st.set_page_config(page_title="Health Advice System", layout="wide")

    # Initialize the health advice system
    health_system = HealthAdviceSystem()

    try:
        # Initialize models
        with st.spinner("Loading models... This may take a moment."):
            health_system.initialize_translation_models()
            health_system.initialize_symptom_analysis()

        # Create two columns for the layout
        col1, col2 = st.columns([2, 1])

        with col1:
            # User input form
            with st.form("health_advice_form"):
                st.subheader("📋 Patient Information")

                age = st.number_input("Age:", min_value=1, max_value=120, help="Enter your age in years")
                weight = st.number_input("Weight (kg):", min_value=1, max_value=500, help="Enter your weight in kilograms")

                st.subheader("📝 Medical History")
                parental_history = st.text_area("Parental Medical History (optional):",
                                                help="Enter any relevant medical conditions in your family")
                personal_history = st.text_area("Personal Medical History:",
                                                help="Enter any previous or current medical conditions")

                st.subheader("🩺 Symptoms")
                symptoms = st.text_input("Symptoms (comma-separated):", help="List the symptoms you are experiencing")

                st.subheader("🔬 Test Results")
                test_results = st.text_area("Test Results (optional):", help="Enter any recent test results")

                submit_button = st.form_submit_button("Get Health Advice")

            if submit_button:
                # Get the health advice
                advice = health_system.get_gpt_advice(age, weight, parental_history, personal_history, symptoms, test_results)
                st.subheader("💡 Health Advice")
                st.write(advice)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
