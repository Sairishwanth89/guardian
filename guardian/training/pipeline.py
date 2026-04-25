"""
GUARDIAN GRPO Pipeline v3
=========================
Production training pipeline. This replaces the flat notebook with a structured
class-based architecture.

Fixes the Critical Weaknesses:
1. Token Wall: Sets max_completion_length=1024 to prevent XML truncation.
2. Stability: Merges reward components dynamically over trajectories.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

@dataclass
class GuardianTrainingArguments:
    model_name: str = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
    run_name: str = "guardian-grpo-v3"
    learning_rate: float = 3e-6
    max_prompt_length: int = 1500
    max_completion_length: int = 1024  # FIX: Truncation wall resolved
    temperature: float = 0.8
    num_generations: int = 4
    max_steps: int = 100
    batch_size: int = 1
    gradient_accumulation_steps: int = 4


class GuardianGRPOPipeline:
    def __init__(self, args: Optional[GuardianTrainingArguments] = None):
        self.args = args or GuardianTrainingArguments()
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
    def _initialize_model(self):
        """Loads model optimally using Unsloth if available."""
        print(f"Loading {self.args.model_name}...")
        try:
            from unsloth import FastLanguageModel
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.args.model_name,
                max_seq_length=2048,
                load_in_4bit=True,
                fast_inference=True,
            )
            # Patch PEFT
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
                lora_alpha=16,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
            )
        except ImportError:
            print("Unsloth not found. Falling back to standard Transformers HuggingFace (Slower).")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.args.model_name,
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
            
    def prepare_dataset(self, samples: List[Dict[str, str]]) -> Dataset:
        """Converts raw prompt/completion pairs into HF Dataset."""
        ds_dict = {"prompt": [], "completion": []}
        for s in samples:
            ds_dict["prompt"].append(s["prompt"])
            ds_dict["completion"].append(s["completion"])
        return Dataset.from_dict(ds_dict)
        
    def execute(self, dataset: Dataset, reward_functions: List[callable]):
        """Runs GRPO using the updated stable configuration."""
        if not self.model:
            self._initialize_model()
            
        training_args = GRPOConfig(
            output_dir=f"./results/{self.args.run_name}",
            run_name=self.args.run_name,
            learning_rate=self.args.learning_rate,
            max_prompt_length=self.args.max_prompt_length,
            max_completion_length=self.args.max_completion_length,
            temperature=self.args.temperature,
            num_generations=self.args.num_generations,
            max_steps=self.args.max_steps,
            per_device_train_batch_size=self.args.batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            logging_steps=1,
            save_steps=20,
            report_to="none"  # Use local logging
        )
        
        self.trainer = GRPOTrainer(
            model=self.model,
            reward_funcs=reward_functions,
            args=training_args,
            train_dataset=dataset,
            processing_class=self.tokenizer
        )
        
        print("Starting GRPO optimization phase...")
        self.trainer.train()
        
    def save(self, path: str):
        if self.trainer:
            self.trainer.save_model(path)
            self.tokenizer.save_pretrained(path)
            print(f"Model saved to {path}")
