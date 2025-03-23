import os
import json
from typing import List, Dict
import openai
from dotenv import load_dotenv
from pydantic import BaseModel
import dspy

# Load environment variables
load_dotenv()

class Speech(BaseModel):
    speaker: str
    text: str
    
class DialogueTurn(BaseModel):
    speaker_a: Speech
    speaker_b: Speech

class NewDialogueGenerator(dspy.Signature):
    """Given past dialogue, generate a list of diverse new turns (at least 10) which should be highly contextually relevant to the past dialogue.
    # Guideline
    1. The new turns should be diverse and creative.
    2. The new turns should utilize the information of the past dialogue
    3. The speech from speaker A cannot directly address the past dialogue, it should be a neutral statement.
    4. The new turns should be natural and conversational.
    """
    
    past_dialogue: List[DialogueTurn] = dspy.InputField(description="The past dialogue turns")
    new_turns: List[str] = dspy.OutputField(description="The new speech of speaker A")

model = "openai/grok-beta"
lm = dspy.LM(
    model=model,
    api_key=os.getenv("API_KEY"),
    api_base=os.getenv("BASE_URL"),
)

dspy.configure(lm=lm)

def generate_variations(x1: str, y1: str, num_variations: int = 1) -> List[Dict[str, str]]:
    generate_data = dspy.Predict(NewDialogueGenerator)
    variations = []
    for _ in range(num_variations):
        new_turns = generate_data(past_dialogue=[DialogueTurn(speaker_a=Speech(speaker="A", text=x1), speaker_b=Speech(speaker="B", text=y1))]).new_turns
        variations.extend(new_turns)
    
    return variations

def save_variations_to_json(variations: List[str], filename: str):
    """Save the variations to a JSON file in instruction-output format"""
    json_data = []
    
    for variation in variations:
        json_data.append({
            "instruction": variation,
            "input": "",  # Empty input as per the template
            "output": ""
        })
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # Example seed dialogue
    seed_x1 = """Hello! I'm Emma, nice to meet you. I'm 28 years old, originally from Portland, Oregon, but I've been living in Austin, Texas for the past three years. Here's a little about me:

Occupation: I work as a software developer for a tech startup focused on environmental sustainability. We develop apps that help people reduce their carbon footprint.

Interests: I'm deeply passionate about technology, environmental conservation, and community building. In my free time, I enjoy:

Hiking and exploring the natural beauty of Texas.
Coding side projects, particularly those that involve AI and machine learning to predict environmental impacts.
Participating in local tech meetups and hackathons.
Volunteering for clean-up drives and educational programs about sustainability.
Background: I studied Computer Science at Oregon State University. My journey into tech was sparked by a high school programming class, and I've been hooked ever since. After college, I worked in Seattle for a couple of years before moving to Austin for the vibrant tech scene and the city's focus on green initiatives.

Personality: I consider myself an introverted extrovert; I love engaging with people, especially about topics I'm passionate about, but I also treasure my alone time to recharge. I'm pretty laid back, always up for a good laugh, and I strive to keep a positive outlook on life.

Goals: I'm aiming to eventually start my own tech company that specializes in sustainable technology solutions. In the shorter term, I want to lead a project at my current job that significantly impacts our approach to waste management.

So, what about you? What brings you here, and what are you interested in?
    """
    seed_y1 = """Hey Emma, it's great to meet you! I'm really impressed by your background and what you're doing with your career. It sounds like you're making a real impact in the tech world, especially in areas that matter for our planet's future.

I'm here just to chat and learn more about interesting people like yourself. My interests are quite varied, but I love learning about technology, especially how it can be used to solve real-world problems. I also enjoy music, reading, and I'm a bit of a foodie.

Your move to Austin sounds like it was a great choice for both your career and personal interests. How have you found the transition from Portland to Austin? And what's the tech scene like there compared to where you've been before?
    """
    
    # Generate variations
    variations = generate_variations(seed_x1, seed_y1, num_variations=1)
    
    # Save results in JSON format
    save_variations_to_json(variations, "/home/azureuser/mnt/rpm/LLaMA-Factory/data/rpm.json")
    print(f"Generated {len(variations)} variations and saved to rpm.json")
