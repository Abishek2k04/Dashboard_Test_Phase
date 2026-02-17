import google.generativeai as genai

# Use your key
genai.configure(api_key="AIzaSyB6klLM-_kwRuq2_Iz822ikbw5vnSqT410")

print("List of available models for this key:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)