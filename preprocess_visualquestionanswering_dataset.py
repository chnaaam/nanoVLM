from datasets import load_dataset

dataset = load_dataset("jp1924/VisualQuestionAnswering")

print(dataset["train"][0])