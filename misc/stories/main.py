import time
from groq import Groq
from tqdm import trange

client = Groq()

for _ in trange(100, desc="Generating"):
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "system",
                "content": "You are a creative writing assistant, tasked with generating engaging and imaginative stories. Your goal is to create stories using simple and common vocabulary. Each story should be captivating, with clear characters, settings, and plots that are easy to understand. Do not include any dialogue. Remember to keep the language straightforward and avoid complex words or concepts.\n\nOnly respond with the story.",
            },
            {
                "role": "user",
                "content": "Please write a long story. The story should use colloquial language. Use simple vocabulary and keep the plot clear and engaging.",
            },
        ],
        temperature=1,
        max_tokens=8192,
        top_p=1,
        stream=False,
        stop=None,
    )

    with open(f"/Users/karan/projects/playground/misc/stories/docs/{hash(completion.choices[0].message.content)}.txt", "w") as f:
        assert completion.choices[0].message.content is not None
        f.write(completion.choices[0].message.content)
        f.close()
