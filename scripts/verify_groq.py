import os
from dotenv import load_dotenv
from groq import Groq

def main():
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ Error: GROQ_API_KEY is not set in environment or .env file.")
        return
    print(f"Testing Groq API (Key preview: {api_key[:10]}...)...")

    client = Groq(api_key=api_key)
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a professional medical assistant."},
            {"role": "user", "content": "Explain briefly in 2 bullet points how aspirin works."}
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=200
    )

    print("\n✅ Groq Response Successful:")
    print(chat_completion.choices[0].message.content)

if __name__ == "__main__":
    main()
