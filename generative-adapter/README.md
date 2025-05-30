# Generative Adapter

This repository contains the code accompanying the paper **"Generative Adapter: Contextualizing Language Models in Parameters with A Single Forward Pass"**. The repository demonstrates how to use our Generative Adapter approach to condition large language models with contextual information in a single forward pass by integrating LoRA (Low-Rank Adaptation) weights.

---

## Repository Structure

- **`requirements.txt`**  
  Lists all the necessary dependencies. **Important:** Make sure to use the following versions for key packages:
  - `transformers==4.42.4`
  - `peft==0.11.1`

- **`inference.ipynb`**  
  A Jupyter Notebook that guides you through the process of:
  - Loading the pretrained base causal language model.
  - Creating a FastLora adapter to inject contextual prompts.
  - Generating conditional text based on a prompt prefix and an input query.

---

## Setup Environment

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your_username/generative-adapter.git
   cd generative-adapter
   ```

2. **Install Dependencies:**

   We recommend using a virtual environment. After setting up your environment, run:

   ```bash
   pip install -r requirements.txt
   ```

   **Note:** Ensure you have the following versions for compatibility:
   - `python==3.10`
   - `transformers==4.42.4`
   - `peft==0.11.1`

---

## Model Inference

1. **Open the Notebook:**
   - Launch Jupyter Notebook or JupyterLab. Or, if you prefer, open it directly in your favorite IDE that supports notebooks.

2. **Execute the Cells:**  
   Run the notebook cells sequentially. The code includes:
   - Loading the base model and tokenizer.
   - Loading the PEFT configuration and FastLora adapter.
   - Generating LoRA weights based on the provided prompt prefix.
   - Generating and printing the final text output.

3. **GPU Requirement:**  
   The notebook is set up to run on a CUDA-enabled GPU (`device = 'cuda'`). Ensure that your system has the necessary GPU support and that your PyTorch installation is configured to use CUDA.

---

## Citation

If you use this code or approach in your research, please consider citing our work:

```bibtex
@misc{chen2024generative,
      title={Generative Adapter: Contextualizing Language Models in Parameters with A Single Forward Pass}, 
      author={Tong Chen and Hao Fang and Patrick Xia and Xiaodong Liu and Benjamin Van Durme and Luke Zettlemoyer and Jianfeng Gao and Hao Cheng},
      year={2024},
      eprint={2411.05877},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2411.05877}, 
}
```
