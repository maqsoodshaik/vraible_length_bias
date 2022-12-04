from datasets import load_dataset, load_metric,concatenate_datasets,Dataset
from transformers import AutoFeatureExtractor
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer
from torch.utils.data import DataLoader
import torch
import numpy as np
import skimage.measure
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

metric = load_metric("accuracy")
model_checkpoint = "/pretrained/wav2vec2-large-xlsr-53en_uscmn_hans_cnfr_fr7.0_3.0_5.0fleurs_bestmodel"
batch_size = 64
feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
max_duration_en_1 = 7.0
max_duration_en_2 = 3.0
max_duration_en_3 = 5.0

dataset_name = "fleurs"
configs = ['en_us']
#,'de_de','nl_nl']
labels =[max_duration_en_1,max_duration_en_2,max_duration_en_3]
#,"German","Dutch"]
label2id, id2label,label2id_int = dict(), dict(),dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label
    label2id_int[label] = i
best_model = AutoModelForAudioClassification.from_pretrained(
    model_checkpoint,output_hidden_states=True
)
def preprocess_function(examples):
    max_duration = max(max_duration_en_1,max_duration_en_2,max_duration_en_3)
    audio_arrays = [x for x in examples["input_values"]]
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=feature_extractor.sampling_rate, 
        max_length=int(feature_extractor.sampling_rate * max_duration), 
        padding="max_length" 
    )
    # inputs["labels"] = [label2id_int[image] for image in examples["language"]]
    return inputs
def preprocess_function_en_1(examples):
    
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=feature_extractor.sampling_rate, 
        max_length=int(feature_extractor.sampling_rate * max_duration_en_1  ), 
        truncation=True
    )
    inputs["labels"] = [label2id_int[max_duration_en_1] for image in examples["language"]]
    return inputs
def preprocess_function_en_2(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=feature_extractor.sampling_rate, 
        max_length=int(feature_extractor.sampling_rate * max_duration_en_2), 
        truncation=True,
        padding=True 
    )
    inputs["labels"] = [label2id_int[max_duration_en_2] for image in examples["language"]]
    return inputs
def preprocess_function_en_3(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=feature_extractor.sampling_rate, 
        max_length=int(feature_extractor.sampling_rate * max_duration_en_3), 
        truncation=True,
        padding=True 
    )
    inputs["labels"] = [label2id_int[max_duration_en_3] for image in examples["language"]]
    return inputs
dataset_validation_en = load_dataset("google/fleurs","en_us",split = "validation")
encoded_dataset_validation_en_1 = dataset_validation_en.map(preprocess_function_en_1,remove_columns=["id","num_samples", "path", "audio", "transcription", "raw_transcription", "gender", "lang_id", "language", "lang_group_id"], batched=True)
encoded_dataset_validation_en_2 = dataset_validation_en.map(preprocess_function_en_2,remove_columns=["id","num_samples", "path", "audio", "transcription", "raw_transcription", "gender", "lang_id", "language", "lang_group_id"], batched=True)
encoded_dataset_validation_en_3 = dataset_validation_en.map(preprocess_function_en_3,remove_columns=["id","num_samples", "path", "audio", "transcription", "raw_transcription", "gender", "lang_id", "language", "lang_group_id"], batched=True)

dataset_validation_combined= concatenate_datasets(
        [encoded_dataset_validation_en_1,encoded_dataset_validation_en_2,encoded_dataset_validation_en_3]
    )
best_model= best_model.wav2vec2
dataset_validation_combined = dataset_validation_combined.map(preprocess_function,batched=True)
dataset_validation_combined.set_format("torch")
eval_dataloader = DataLoader(dataset_validation_combined, batch_size=16)
pred = torch.tensor([])
best_model = best_model.to(device)
labels_p= torch.tensor([])
domain= torch.tensor([])
best_model.eval()
# breakpoint()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = best_model(batch["input_values"])
        pred_s = outputs.last_hidden_state.to("cpu")
        pred = torch.cat((pred,pred_s),0)
        labels_s = batch["labels"].to("cpu")
        labels_p = torch.cat((labels_p,labels_s),0)
        # domain_s =  batch["domain"].to("cpu")
        # domain = torch.cat((domain,domain_s),0)
torch.save(labels_p, f'feature_extracted/{model_checkpoint.split("/")[-1]}_{configs[0]}_length.pt') 
torch.save(pred,f'feature_extracted/{model_checkpoint.split("/")[-1]}last_hidden_states.pt')
print("Done!")