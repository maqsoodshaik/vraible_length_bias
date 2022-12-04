from datasets import load_dataset, load_metric,concatenate_datasets,Dataset
from transformers import AutoFeatureExtractor
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer

from torch.utils.data import DataLoader
import torch
import numpy as np
from typing import Optional, Tuple
import skimage.measure
#import weights and bias
import wandb
#device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wandb.init(project='variable_length_bias')
config = wandb.config
# P = np.load("P_100_EN_length.npy")
P = np.load("P_100_EN_wav2vec2-large-xlsr-53en_uscmn_hans_cnfr_fr7_length.npy")
metric = load_metric("accuracy")
metric_f1 = load_metric("f1")
model_checkpoint = "/pretrained/wav2vec2-large-xlsr-53en_uscmn_hans_cnfr_fr7.0_3.0_5.0fleurs_bestmodel"
batch_size = 16
feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
max_duration = 10.0  # seconds
config.max_duration_en = 3.0
config.max_duration_fr = 7.0
config.max_duration_cn = 5.0

dataset_name = "fleurs"
configs = ['en_us','cmn_hans_cn','fr_fr']
#,'de_de','nl_nl']
labels =["English","Mandarin Chinese","French"]
#,"German","Dutch"]
model_name_extension = "".join(configs)
model_name = model_checkpoint.split("/")[-1]+model_name_extension+dataset_name
label2id, id2label,label2id_int = dict(), dict(),dict()

for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label
    label2id_int[label] = i
best_model = AutoModelForAudioClassification.from_pretrained(
    model_checkpoint, num_labels=len(labels),label2id=label2id,
    id2label=id2label)

class Model_with_new_head(torch.nn.Module):
    def __init__(self, model, new_head):
        super(Model_with_new_head, self).__init__()
        self.model_wav = model.wav2vec2
        self.model_proj = model.projector
        self.model_class = model.classifier
        self.new_head = new_head

    def forward(self,x):
        outputs = self.model_wav(torch.stack( x["input_values"]).permute(1,0).to(device).type(torch.float32))
        # inter = outputs[0].dot(self.new_head)
        # #dot prodcut of 3d tensor with 2d tensor
        inter = torch.einsum('ijk,kl->ijl', outputs[0], self.new_head)
        
        inter = self.model_proj(inter)
        loss = None
        logits = self.model_class(inter.mean(dim=1))
        return logits
org_model = best_model
best_model = Model_with_new_head(best_model,torch.nn.Parameter(torch.from_numpy(P).float()))
model_name_extension = "".join(configs)
model_name = model_checkpoint.split("/")[-1]+model_name_extension+dataset_name
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
def compute_metrics_f1(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric_f1.compute(predictions=predictions, references=eval_pred.label_ids,average="weighted")


dataset_validation_en = load_dataset("google/fleurs","en_us",split = "validation")
dataset_validation_cn = load_dataset("google/fleurs","cmn_hans_cn",split = "validation")
dataset_validation_fr = load_dataset("google/fleurs","fr_fr",split = "validation")
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
encoded_dataset_fr = dataset_validation_fr.map(preprocess_function_fr,remove_columns=["id","num_samples", "path", "audio", "transcription", "raw_transcription", "gender", "lang_id", "language", "lang_group_id"], batched=True)
encoded_dataset_en = dataset_validation_en.map(preprocess_function_en, remove_columns=["id","num_samples", "path", "audio", "transcription", "raw_transcription", "gender", "lang_id", "language", "lang_group_id"], batched=True)
encoded_dataset_cn = dataset_validation_cn.map(preprocess_function_cn, remove_columns=["id","num_samples", "path", "audio", "transcription", "raw_transcription", "gender", "lang_id", "language", "lang_group_id"], batched=True)
dataset_validation_combined= concatenate_datasets(
        [encoded_dataset_en,encoded_dataset_cn,encoded_dataset_fr]
    )
encoded_dataset_validation = dataset_validation_combined.map(preprocess_function,batched=True)

#create dataloader from encoded_dataset_validation
dataloader = DataLoader(encoded_dataset_validation, batch_size=batch_size, shuffle=False)
#calculate accuracy and f1 score on validation set using dataloader
trainer = Trainer(
    org_model,
    args,
    eval_dataset=encoded_dataset_validation,
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics
)
trainer_f1 = Trainer(
    org_model,
    args,
    tokenizer=feature_extractor,
    eval_dataset=encoded_dataset_validation,
    compute_metrics=compute_metrics_f1
)
def calculate_accuracy_f1(model,dataloader):
    model = model.to(device)
    model.eval()
    predictions = []
    labels = []
    for batch in dataloader:
        with torch.no_grad():
            logits = model(batch)
            predictions.extend(torch.argmax(logits, axis=1).tolist())
            labels.extend(batch["labels"].tolist())
    return metric.compute(predictions=predictions, references=labels),metric_f1.compute(predictions=predictions, references=labels,average="weighted")
print(calculate_accuracy_f1(best_model,dataloader))
print(trainer.evaluate())
print("F1")
print(trainer_f1.evaluate())