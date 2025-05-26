import json
import random

input_path = 'data.jsonl'
output_path = 'sampled_500.jsonl'
sample_size = 500

# Load all data points
with open(input_path, 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

# Sample 500 random data points
sampled = random.sample(data, min(sample_size, len(data)))

# Save to new jsonl file
with open(output_path, 'w', encoding='utf-8') as f:
    for item in sampled:
        f.write(json.dumps(item) + '\n')