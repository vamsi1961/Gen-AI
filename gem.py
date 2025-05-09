from google_genai import GeminiClient

# Initialize the client
client = GeminiClient(api_key='YOUR_API_KEY')

# Make a request
response = client.generateContent(prompt="Hello, Gemini!")
print(response)
