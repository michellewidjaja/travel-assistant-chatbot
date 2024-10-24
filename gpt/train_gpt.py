from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import json

with open('travel_dataset.json', 'r') as f:
    data = json.load(f)

inputs = []
responses = []
for entry in data:
    if isinstance(entry["input"], list):
        for input_text in entry["input"]:
            inputs.append(input_text)
            responses.append(entry["response"] if isinstance(entry["response"], str) else entry["response"][0])
    else:
        inputs.append(entry["input"])
        responses.append(entry["response"] if isinstance(entry["response"], str) else entry["response"][0])

dataset_dict = {
    "input": inputs,
    "response": responses
}

dataset = Dataset.from_dict(dataset_dict)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    inputs = [f"Input: {q} Response: {a}" for q, a in zip(examples["input"], examples["response"])]
    tokenized_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=256)
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    gradient_accumulation_steps=4,
    logging_dir='./logs',
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
)

trainer.train()

model.save_pretrained("fine_tuned_models")
tokenizer.save_pretrained("fine_tuned_models")
