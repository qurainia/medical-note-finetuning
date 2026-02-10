# process the hf data from https://huggingface.co/datasets/AGBonnet/augmented-clinical-notes
# into a format that can be used by NeMo Optimizer
import json

data_out  = []
in_file_path = 'augmented_notes_30K.jsonl'
out_file_prefix = 'data_for_optimizer_'
train, val, test = 1000, 100, 100

instruction = "You are a helpful assistant that can generate a clinical note from a conversation transcript between a patient and a doctor. \n\n"

def reformat_json_object(json_object):
    return {
        "prompt": [
            {"role": "system", "content": "detailed thinking off\n" + instruction },
            {"role": "user", "content": "The conversation transcript is given below: \n\n" + json_object["conversation"]}
        ],
        "completion": json_object["note"]
    }

count = 0
# Open the file in read mode and iterate through each line
with open(in_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        count += 1
        if count % 100 == 0:
            print(f"Processed {count} records")
        if count > train+val+test:
            break
        # Parse each line as a JSON object (dictionary) and append to the list
        try:
            json_object = json.loads(line.strip())
            reformatted_json_object = reformat_json_object(json_object)
            data_out.append(reformatted_json_object)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON on line: {e}")

# Now 'data' is a list of dictionaries, ready for further use
print(f"Read {len(data_out)} records.")




# Open the file in write mode ('w')
with open(out_file_prefix + 'train.jsonl', 'w') as outfile:
    # Iterate over the list of Python objects
    for entry in data_out[:train]:
        # Convert each object to a JSON string and write it as a new line
        json_string = json.dumps(entry)
        outfile.write(json_string + '\n')

with open(out_file_prefix + 'val.jsonl', 'w') as outfile:
    # Iterate over the list of Python objects
    for entry in data_out[train:train+val]:
        # Convert each object to a JSON string and write it as a new line
        json_string = json.dumps(entry)
        outfile.write(json_string + '\n')

with open(out_file_prefix + 'test.jsonl', 'w') as outfile:
    # Iterate over the list of Python objects
    for entry in data_out[train+val:train+val+test]:
        # Convert each object to a JSON string and write it as a new line
        json_string = json.dumps(entry)
        outfile.write(json_string + '\n')