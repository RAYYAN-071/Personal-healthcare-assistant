import openai

# Your API key
openai.api_key = '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'

def get_gpt_advice(age, weight, parental_history, personal_history, symptoms, test_results):
    """Get medical advice from GPT-3.5-turbo with error handling"""
    try:
        prompt = f"""
        Provide personalized health advice based on the following information:
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
        6. Explanation of medical terms in simple language
        """

        response = openai.completions.create(
            model="gpt-3.5-turbo",
            prompt=prompt,
            max_tokens=1000,
            temperature=0.7
        )
        return response['choices'][0]['text'].strip()
    except Exception as e:
        print(f"Error getting GPT advice: {e}")
        return None
