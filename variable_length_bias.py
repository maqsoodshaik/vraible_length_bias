from datasets import load_dataset, load_metric,concatenate_datasets,Dataset
from transformers import AutoFeatureExtractor
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer
from torch.utils.data import DataLoader
import torch
import numpy as np
import skimage.measure
#import weights and bias
import wandb
import torch
torch.cuda.empty_cache()
#set seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)
wandb.init(project='variable_length_bias')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = wandb.config
# config.max_duration_en = 7.0
# config.max_duration_cn = 3.0
# config.max_duration_fr = 5.0
config.max_duration_en = 7.0
config.max_duration_cn = 3.0
config.max_duration_fr = 5.0
metric = load_metric("accuracy")
model_checkpoint = "facebook/wav2vec2-base"
batch_size = 64
feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
max_duration = 10.0  # seconds

dataset_name = "fleurs"
configs = ['en_us','cmn_hans_cn','fr_fr']
#,'de_de','nl_nl']
labels =["English","Mandarin Chinese","French"]
#,"German","Dutch"]

label2id, id2label,label2id_int = dict(), dict(),dict()

for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label
    label2id_int[label] = i
best_model = AutoModelForAudioClassification.from_pretrained(
    model_checkpoint, num_labels=len(labels),label2id=label2id,
    id2label=id2label)
model_name_extension = "".join(configs)
model_name_time = "_".join([str(config.max_duration_en),str(config.max_duration_cn),str(config.max_duration_fr)])
model_name = model_checkpoint.split("/")[-1]+model_name_extension+model_name_time+dataset_name
args = TrainingArguments(
    f"{model_name}",#{model_name}arnlpt
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=2,
    warmup_ratio=0.1,
    logging_steps=10,
    eval_accumulation_steps = 1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)
def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


dataset_train_en = load_dataset("google/fleurs","en_us",split = "train")
dataset_train_cn = load_dataset("google/fleurs","cmn_hans_cn",split = "train")
dataset_train_fr = load_dataset("google/fleurs","fr_fr",split = "train")
dataset_test_en = load_dataset("google/fleurs","en_us",split = "test")
dataset_test_cn = load_dataset("google/fleurs","cmn_hans_cn",split = "test")
dataset_test_fr = load_dataset("google/fleurs","fr_fr",split = "test")
def preprocess_function(examples):
    max_duration = max(config.max_duration_en,config.max_duration_cn,config.max_duration_fr)
    audio_arrays = [x for x in examples["input_values"]]
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=feature_extractor.sampling_rate, 
        max_length=int(feature_extractor.sampling_rate * max_duration), 
        padding="max_length" 
    )
    # inputs["labels"] = [label2id_int[image] for image in examples["language"]]
    return inputs

def preprocess_function_en(examples):
    
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=feature_extractor.sampling_rate, 
        max_length=int(feature_extractor.sampling_rate * config.max_duration_en ), 
        truncation=True
    )
    inputs["labels"] = [label2id_int[image] for image in examples["language"]]
    return inputs
def preprocess_function_fr(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=feature_extractor.sampling_rate, 
        max_length=int(feature_extractor.sampling_rate * config.max_duration_fr), 
        truncation=True
    )
    inputs["labels"] = [label2id_int[image] for image in examples["language"]]
    return inputs
def preprocess_function_cn(examples):
    
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=feature_extractor.sampling_rate, 
        max_length=int(feature_extractor.sampling_rate * config.max_duration_cn), 
        truncation=True
    )
    inputs["labels"] = [label2id_int[image] for image in examples["language"]]
    return inputs
encoded_dataset_fr = dataset_train_fr.map(preprocess_function_fr,remove_columns=["id","num_samples", "path", "audio", "transcription", "raw_transcription", "gender", "lang_id", "language", "lang_group_id"], batched=True)
encoded_dataset_en = dataset_train_en.map(preprocess_function_en, remove_columns=["id","num_samples", "path", "audio", "transcription", "raw_transcription", "gender", "lang_id", "language", "lang_group_id"], batched=True)
encoded_dataset_cn = dataset_train_cn.map(preprocess_function_cn, remove_columns=["id","num_samples", "path", "audio", "transcription", "raw_transcription", "gender", "lang_id", "language", "lang_group_id"], batched=True)
encoded_dataset_test_fr = dataset_test_fr.map(preprocess_function_fr,remove_columns=["id","num_samples", "path", "audio", "transcription", "raw_transcription", "gender", "lang_id", "language", "lang_group_id"], batched=True)
encoded_dataset_test_en = dataset_test_en.map(preprocess_function_en, remove_columns=["id","num_samples", "path", "audio", "transcription", "raw_transcription", "gender", "lang_id", "language", "lang_group_id"], batched=True)
encoded_dataset_test_cn = dataset_test_cn.map(preprocess_function_cn, remove_columns=["id","num_samples", "path", "audio", "transcription", "raw_transcription", "gender", "lang_id", "language", "lang_group_id"], batched=True)
# import gc
# del dataset_train_en
# del dataset_train_cn
# del dataset_train_fr
# del dataset_test_en
# del dataset_test_cn
# del dataset_test_fr
dataset_train_combined= concatenate_datasets(
        [encoded_dataset_en,encoded_dataset_cn,encoded_dataset_fr]
    )
# del encoded_dataset_en
# del encoded_dataset_cn
# del encoded_dataset_fr
dataset_test_combined= concatenate_datasets(
        [encoded_dataset_test_en,encoded_dataset_test_cn,encoded_dataset_test_fr]
    )
# del encoded_dataset_test_en
# del encoded_dataset_test_cn
# del encoded_dataset_test_fr
encoded_dataset_train = dataset_train_combined.map(preprocess_function,batched=True)
encoded_dataset_test = dataset_test_combined.map(preprocess_function,batched=True)

# del dataset_train_combined
# del dataset_test_combined
# gc.collect()
# encoded_dataset_train.set_format("torch")
# encoded_dataset_test.set_format("torch")
#split train set into train and validation set using 80/20 split
# train_dataset, validation_dataset = train_test_split(train_dataset, test_size=0.2, random_state=42)
# train_dataset.set_format("torch")
# validation_dataset.set_format("torch")
# eval_dataloader = DataLoader(encoded_dataset_test, batch_size=16)
# train_dataloader = DataLoader(encoded_dataset_train, batch_size=16)
# test_dataloader = DataLoader(test_dataset, batch_size=16)
trainer = Trainer(
    best_model,
    args,
    train_dataset=encoded_dataset_train,
    eval_dataset=encoded_dataset_test,
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics
)

print(trainer.train())
trainer.save_model( f"/pretrained/{model_name}_bestmodel")
print(trainer.evaluate())

