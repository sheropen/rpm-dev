import os
import json
import logging
from typing import List, Dict
import openai
from dotenv import load_dotenv
from pydantic import BaseModel
import dspy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class QAPair(BaseModel):
    question: str
    answer: str

class NewDialogueGenerator(dspy.Signature):
    """Given a passage, generate a list of diverse qa pairs that utilize the information of the passage.
    # Guideline
    1. The generated qa pairs should cover every information in the passage.
    2. The generated qa pairs should be diverse.
    3. Generate as much qa pairs as possible.
    """
    
    passage: str = dspy.InputField(description="The passage")
    qa_pairs: List[QAPair] = dspy.OutputField(description="The generated qa pairs")

model = "openai/grok-beta"
lm = dspy.LM(
    model=model,
    api_key=os.getenv("API_KEY"),
    api_base=os.getenv("BASE_URL"),
    temperature=0.9
)

dspy.configure(lm=lm)

def get_golden_passages(data_entry: Dict) -> List[str]:
    logger.info("Extracting golden passages from data entry")
    all_passages = {title: " ".join(content) for title, content in data_entry['context']}
    golden_title = set([x[0] for x in data_entry['supporting_facts']])
    golden_passages = {title: all_passages[title] for title in golden_title}
    golden_passages = list(golden_passages.values())
    logger.info(f"Extracted {len(golden_passages)} golden passages")
    return golden_passages

def generate_qas(passages: List[str], num_variations: int = 3) -> List[Dict[str, str]]:
    logger.info(f"Generating QA pairs from {len(passages)} passages")
    generate_data = dspy.Predict(NewDialogueGenerator)
    qa_pairs = []
    for i, passage in enumerate(passages):
        logger.info(f"Processing passage {i+1}/{len(passages)}")
        for _ in range(num_variations):
            result = generate_data(passage=passage)
            qa_pairs.extend(result.qa_pairs)
            logger.info(f"Generated {len(result.qa_pairs)} QA pairs from passage {i+1}")
    
    logger.info(f"Total QA pairs generated: {len(qa_pairs)}")
    return qa_pairs

def save_qas_to_json(qa_pairs: List[QAPair], filename: str):
    """Save the qa pairs to a JSON file in instruction-output format"""
    logger.info(f"Saving {len(qa_pairs)} QA pairs to {filename}")
    json_data = []
    
    for qa_pair in qa_pairs:
        json_data.append({
            "instruction": qa_pair.question,
            "input": "",  # Empty input as per the template
            "output": qa_pair.answer
        })
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
    logger.info(f"Successfully saved QA pairs to {filename}")

def save_metadata_to_json(data: Dict, passages: List[str], filename: str):
    """Save the metadata to a JSON file"""
    logger.info(f"Saving metadata to {filename}")
    metadata = data
    metadata['passages'] = passages
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    logger.info("Starting HotpotQA data generation")
    # Load the passage
    data_path = "/home/azureuser/mnt/rpm/PRAG/data/hotpotqa/hotpot_dev_distractor_v1.json"
    logger.info(f"Loading dataset from {data_path}")
    with open(data_path, 'r') as f:
        dataset = json.load(f)
    logger.info(f"Loaded dataset with {len(dataset)} entries")
    
    
    # Generate variations
    data = dataset[0]
    logger.info("Extracting golden passages from first dataset entry")
    passages = get_golden_passages(data)
    
    # Split passages if it's a single string
    if isinstance(passages, str):
        passages = [passages]
    
    logger.info("Generating QA pairs")
    qa_pairs = generate_qas(passages)
    
    # Save results in JSON format
    output_dir = f"/home/azureuser/mnt/rpm/LLaMA-Factory/data/rpm/hotpot_qa/{data['_id']}"
    os.makedirs(output_dir, exist_ok=True)
    qa_output_path = os.path.join(output_dir, "qa_pairs.json")
    logger.info(f"Saving results to {qa_output_path}")
    save_qas_to_json(qa_pairs, qa_output_path)
    save_metadata_to_json(data, passages, os.path.join(output_dir, "metadata.json"))
    logger.info(f"Generated {len(qa_pairs)} qa pairs and saved to rpm_hotpot_qa.json")
    logger.info("Process completed successfully")
