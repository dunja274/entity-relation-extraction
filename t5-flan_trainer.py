import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"


from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import evaluate
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
nltk.download("punkt")

# model_id="google/flan-t5-base"
# model_id="google/mt5-base"


tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-base")
df = pd.read_json('/mnt/internships/2023-relation-extraction-and-grouping/data/fee-notifications-plus-dataset.json')

# Define a function to extract entity groups from annotated_entity_groups column
def extract_entity_groups(annotated_entity_groups):
    entity_groups = []
    if len(annotated_entity_groups):
        for group in annotated_entity_groups[0]['groups']:
            group_id = group['id']
            entity_ids = group['entity_ids']
            entity_groups.append({'id': group_id, 'entity_ids': entity_ids})
    return entity_groups

# Define a function to create input-output pairs
def create_input_output_pairs(df):
    input_output_pairs = []
    for index, row in df.iterrows():
        input_dict = {
            'document': row['document'],
            'annotated_entities': row['annotated_entities']
        }
        output_list = extract_entity_groups(row['annotated_entity_groups'])
        input_output_pairs.append((input_dict, output_list))
    return input_output_pairs

# Create input-output pairs
input_output_pairs = create_input_output_pairs(df)



for index, row in df.iterrows():
    # Uupdate group values with generated groups
    entity_groups = extract_entity_groups(row['annotated_entity_groups'])
    df.at[index, 'annotated_entity_groups'] = [entity_group['entity_ids'] for entity_group in entity_groups]


# replace entity ids with tokens

token_groups = []
df['group_ids']=df['annotated_entity_groups']
for index, row in df.iterrows():
    entities = row['annotated_entities']
    zero_id = row['token_ids']
    entity_groups = row['annotated_entity_groups']
    new_group = []
    new_tokens =[]
    for group in entity_groups:
        tmp=[]
        tmp2=[]
        for g in group:
            for en in entities:
                for e in en['entities']:
                    if e['id']==g:
                        tmp.append(e['text'])
                        tmp2.append(e['token_ids'])
        new_group.append(tmp)
        new_tokens.append(tmp2)
        token_groups.append(new_tokens)
    df.at[index, 'annotated_entity_groups'] = new_group
    df.at[index,'group_ids'] = new_tokens
    

df.drop('annotated_entity_relationships', axis=1, inplace=True)


# extract entity positions

tag_groups = []
for index, row in df.iterrows():
    doc_groups = []
    # Conditionally update values in the 'label' column
    entities = row['annotated_entities']
    first = row["token_ids"][0]
    for en in entities:
        # print(en)
        for e in en['entities']:
            doc_groups.append([x - first for x in e['token_ids']])
    tag_groups.append(doc_groups)

# print(tag_groups[0])
for i,group in enumerate(tag_groups):
    tag_groups[i] = sorted(group, key=lambda sublist: sublist[0])


# add entity tags

def insert_tags(indexes, my_list):
    
    adjustment = 0  # Tracks the cumulative adjustment for indexes
    i=0
    for sublist in indexes:
        first_index = sublist[0] + adjustment
        last_index = sublist[-1] + adjustment
        # Adjust indexes based on previous insertions
        
        # Insert item at the adjusted indexes
        my_list.insert(first_index, f"<e{i}>")
        my_list.insert(last_index + 2, f"</e{i}>")
        i+=1
        adjustment += 2  # Update the adjustment value

    return my_list


def apply_insert_tags(row):
    index = row.name
    group = tag_groups[index]
    tokens = row['tokens']
    modified_tokens = insert_tags(group, list(tokens))
    row['tokens'] = modified_tokens
    return row

# df = df.apply(apply_insert_tags, axis=1) ## adding tags


# Define a preprocessing function
def preprocess(row, tokenizer):
    # Tokenize the text
    tokens = row["tokens"]

    tokenized_tokens = tokenizer(tokens, is_split_into_words=True,  padding='max_length', truncation=True, max_length=512)
    tokenized_labels = tokenizer(str(row['annotated_entity_groups']), is_split_into_words=False,  padding='max_length', truncation=True, max_length=60) #80
    
    # Get the token IDs
    input_ids = tokenized_tokens['input_ids']
    attention_mask = tokenized_tokens['attention_mask']
    tokenized_labels["input_ids"] = [(l if l != tokenizer.pad_token_id else -100) for l in tokenized_labels["input_ids"]] 

    preprocessed_data = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        'labels':tokenized_labels['input_ids']
    }

    return preprocessed_data

# Apply the preprocessing function to each row in the DataFrame
# df_preprocessed = df.apply(preprocess, axis=1, result_type="expand", tokenizer=tokenizer)





# Split into training and validation subsets
train_df, val_df = train_test_split(df, test_size=0.3, random_state=42) 
val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=42)

# train set size experiments
# train_df = train_df.head(2000)

train_df = train_df.apply(preprocess, axis=1, result_type="expand", tokenizer=tokenizer)
val_df = val_df.apply(preprocess, axis=1, result_type="expand", tokenizer=tokenizer)
test_df = test_df.apply(preprocess, axis=1, result_type="expand", tokenizer=tokenizer)



# Create a custom dataset using the 'df_preprocessed' DataFrame
def create_custom_dataset(df):
    dataset = Dataset.from_pandas(df)
    dataset_dict = {col: dataset[col] for col in dataset.column_names}
    dataset = Dataset.from_dict(dataset_dict)
    return dataset


train_dataset = create_custom_dataset(train_df)
valid_dataset = create_custom_dataset(val_df)
test_dataset = create_custom_dataset(test_df)

def preprocess_function(sample,padding="max_length"):
    return sample

tokenized_train = train_dataset.map(preprocess_function, batched=True)

tokenized_valid = valid_dataset.map(preprocess_function, batched=True)
tokenized_test = test_dataset.map(preprocess_function, batched=True)


# Metric
metric = evaluate.load("rouge")

# helper function to postprocess text
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    outputs = {
        'prediction':decoded_preds,
        'label':decoded_labels
    }

    # Open a file for writing
    # with open('outputs.json', 'w') as f:
    #     # Serialize the dictionary to a JSON string and write it to the file
    #     json.dump(outputs, f)

    # print(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result


# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)


# Define training args
training_args = Seq2SeqTrainingArguments(
    output_dir='./dipl_testing/t5-base-2000',
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    predict_with_generate=True,
    fp16=False, # Overflows with fp16
    learning_rate=5e-5,
    num_train_epochs=10,
    # logging & evaluation strategies
    logging_dir=f"./dipl_testing/logs",
    logging_strategy="steps",
    logging_steps=500,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    # metric_for_best_model="overall_f1",
    push_to_hub=False,
    # gradient_accumulation_steps=4,
)


# model = AutoModelForSeq2SeqLM.from_pretrained('./dipl_testing/t5-base-2000/checkpoint-10000')

# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    compute_metrics=compute_metrics,
)

trainer.train()

print('valid')
out = trainer.evaluate(tokenized_valid)

print(out)
print()

print('test')
out = trainer.evaluate(tokenized_test)

print(out)