from datasets import get_dataset_config_names
import pickle
import torch
import shutil
from torch.utils.data import DataLoader
import os
from transformers import AutoFeatureExtractor, Wav2Vec2Model
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices
max_duration = 10.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from datasets import load_dataset, concatenate_datasets

feature_extractor = AutoFeatureExtractor.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53"
)
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53").to(
    device
)
configs = get_dataset_config_names("google/fleurs")
configs = ["en_us","nl_nl","de_de","fr_fr","es_419"]
def preprocess_function_f(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=feature_extractor.sampling_rate, 
        max_length=int(feature_extractor.sampling_rate * max_duration), 
        truncation=True,
        padding=True 
    )
    return inputs
print(configs)
for name in configs:
    ds_train = load_dataset("google/fleurs", name,split="train")
    ds_validation = load_dataset("google/fleurs", name, split="validation")
    ds_test = load_dataset("google/fleurs", name, split="test")

    ds = concatenate_datasets(
        [ds_train,ds_validation,ds_test]
    )  # ,ds_validation,ds_test,ds_train
    print(len(ds))
    ds = ds.map(preprocess_function_f, remove_columns=["id","num_samples", "path", "audio", "transcription", "raw_transcription",  "lang_id", "language", "lang_group_id"], batched=True)

    ds_validation = []
    ds_test = []
    projected_states = torch.tensor([])
    projected_quantized_states = torch.tensor([])
    hidden_states = torch.tensor([])
    gender = torch.tensor([])
    last_hidden_states =torch.tensor([])
    raw_sequence_length_m =0
    ds.set_format("torch")
    eval_dataloader = DataLoader(ds, batch_size=16)

    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(batch["input_values"])
        
        last_hidden_states = torch.cat((last_hidden_states,outputs.last_hidden_state.to("cpu")),dim=0)
        gender = torch.cat((gender,batch["gender"].to("cpu")),dim=0)

    # torch.save(projected_states,f'feature_extracted/{name}projected_states_all.pt')
    # torch.save(projected_quantized_states,f'feature_extracted/{name}projected_quantized_states_all.pt')
    torch.save(gender, f'feature_extracted/{name}gender.pt') 
    torch.save(last_hidden_states,f'feature_extracted/{name}last_hidden_states.pt')
    print("Done!")

