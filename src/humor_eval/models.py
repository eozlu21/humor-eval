# models.py
from transformers import AutoProcessor, Glm4vForConditionalGeneration
import torch
from PIL.Image import Image

MODEL_ID = "THUDM/GLM-4.1V-9B-Thinking"

def load_model():
    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)
    model = Glm4vForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",   # sharded
    )
    model.eval()
    return processor, model

def chat_infer(processor: AutoProcessor, model: Glm4vForConditionalGeneration,
               image: Image, text: str, max_new_tokens: int = 64) -> str:
    messages = [
        {"role": "user",
         "content": [{"type": "image", "image": image},
                     {"type": "text", "text": text}]}
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt", 
    ).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]).strip()
