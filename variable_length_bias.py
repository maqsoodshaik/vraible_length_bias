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
    # Set the random seed manually for reproducibility.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
set_seed(42)
wandb.init(project='variable_length_bias')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = wandb.config
config.max_duration_en = 7.0
config.max_duration_cn = 3.0
config.max_duration_fr = 5.0
config.model = "xlsr"
# config.max_duration_en = 3.0
# config.max_duration_cn = 3.0
# config.max_duration_fr = 3.0
metric = load_metric("accuracy")
model_checkpoint = "facebook/wav2vec2-large-xlsr-53"
batch_size = 16
feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)

dataset_name = "fleurs"
configs = ['en_us','cmn_hans_cn','fr_fr']

labels =["English","Mandarin Chinese","French"]


label2id, id2label,label2id_int = dict(), dict(),dict()

for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label
    label2id_int[label] = i
best_model = AutoModelForAudioClassification.from_pretrained(
    model_checkpoint, num_labels=len(labels),label2id=label2id,
    id2label=id2label,output_hidden_states=True)
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
class Load_dataset():
    def __init__(self,dataset_name,configs,language_duration,feature_extractor,label2id_int):
        self.dataset_name = dataset_name
        self.configs = configs
        self.language_duration = language_duration
        self.feature_extractor = feature_extractor
        self.label2id_int = label2id_int
    
    def preprocess_function_language(self,examples):
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = self.feature_extractor(
        audio_arrays, 
        sampling_rate=self.feature_extractor.sampling_rate, 
        max_length=int(self.feature_extractor.sampling_rate *self.language_duration), 
        truncation=True
        )
        inputs["labels"] = [label2id_int[image] for image in examples["language"]]
        return inputs
    def train_split(self):
        dataset_loaded_train = load_dataset(self.dataset_name, self.configs, split="train")
        
        encoded_dataset_train = dataset_loaded_train.map(self.preprocess_function_language,remove_columns=["id","num_samples", "path", "audio", "transcription", "raw_transcription", "gender", "lang_id", "language", "lang_group_id"], batched=True)
        
        return encoded_dataset_train
    def test_split(self):
        dataset_loaded_test = load_dataset(self.dataset_name, self.configs, split="test")
        encoded_dataset_test = dataset_loaded_test.map(self.preprocess_function_language,remove_columns=["id","num_samples", "path", "audio", "transcription", "raw_transcription", "gender", "lang_id", "language", "lang_group_id"], batched=True)
        return encoded_dataset_test
def preprocess_function(examples):
        max_duration = max(config.max_duration_en,config.max_duration_cn,config.max_duration_fr)
        audio_arrays = [x for x in examples["input_values"]]
        inputs = feature_extractor(
            audio_arrays, 
            sampling_rate=feature_extractor.sampling_rate, 
            max_length=int(feature_extractor.sampling_rate * max_duration), 
            padding="max_length" 
        )
        return inputs
train_dataset_list = []
test_dataset_list = []
for config_lang,duration in zip(configs,[config.max_duration_en,config.max_duration_cn,config.max_duration_fr]):
    dataset_obj = Load_dataset("google/fleurs",config_lang,duration,feature_extractor,label2id_int)
    train_dataset_list.append(dataset_obj.train_split())
    test_dataset_list.append(dataset_obj.test_split())
dataset_train_combined= concatenate_datasets(
        train_dataset_list
    )

dataset_test_combined= concatenate_datasets(
        test_dataset_list
    )
equal_length = []
for config_lang_same,duration_same in zip(configs,[7.0,7.0,7.0]):
    dataset_obj_same = Load_dataset("google/fleurs",config_lang_same,duration_same,feature_extractor,label2id_int)
    equal_length.append(dataset_obj_same.test_split())
dataset_test_equal_length= concatenate_datasets(
        equal_length
    )
encoded_dataset_train = dataset_train_combined.map(preprocess_function,batched=True)
encoded_dataset_test = dataset_test_combined.map(preprocess_function,batched=True)
encoded_dataset_test_equal_length = dataset_test_equal_length.map(preprocess_function,batched=True)

encoded_dataset_train.set_format("torch")
encoded_dataset_test.set_format("torch")
encoded_dataset_test_equal_length.set_format("torch")
eval_dataloader = DataLoader(encoded_dataset_test, batch_size=16)
eval_dataloader_hidden = DataLoader(encoded_dataset_test_equal_length, batch_size=16,shuffle=True,drop_last=True)
train_dataloader = DataLoader(encoded_dataset_train, batch_size=16,shuffle=True,drop_last=True)


config.epochs = 2
from transformers import get_linear_schedule_with_warmup, AdamW
optimizer = AdamW(best_model.parameters(), lr=3e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataloader)*config.epochs*0.1, num_training_steps=len(train_dataloader)*config.epochs)
best_model = best_model.to(device)
#accuracy_score from logits
from sklearn.linear_model import LogisticRegression
#import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
criterion = torch.nn.CrossEntropyLoss()
def accuracy_score(preds, labels):
    preds = np.argmax(preds, axis=1)
    return (preds == labels).mean()
def eval_func(eval_dataloader, best_model):
    n_correct = 0
    n_samples = 0
    for batch in eval_dataloader:
        best_model.eval()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = best_model(batch["input_values"])
        loss = outputs.loss
        _, predicted = torch.max(outputs.logits, 1)
        n_samples += 1
        n_correct += (predicted == batch["labels"]).sum().item()
    acc = 100.0 * n_correct / len(encoded_dataset_test)
    print(f'Accuracy of the network on the validation images: {acc} %')
    #log to wandb
    wandb.log({"Accuracy on validation": acc})
def eval_hidden_func(eval_dataloader_hidden, best_model):
    labels_p= torch.tensor([])
    d = {}
    for x in range(13):
        d[f"hidden_state_{x}"] = torch.tensor([])
    for batch in eval_dataloader_hidden:
        best_model.eval()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = best_model(batch["input_values"])
        labels_s = batch["labels"].to("cpu")
        labels_p = torch.cat((labels_p,labels_s),0)
        for num,hidden_state in enumerate(outputs.hidden_states):
                #LogisticRegression from sklearn to predict the language
                hidden_state = hidden_state.to("cpu")
                hidden_state = hidden_state.mean(dim=1)
                hidden_state = hidden_state.reshape(hidden_state.shape[0], -1)
                d[f"hidden_state_{num}"] = torch.cat((d[f"hidden_state_{num}"],hidden_state),0)
    for v in range(13):
        X_train, X_test, y_train, y_test = train_test_split(d[f"hidden_state_{v}"], labels_p, test_size=0.2, random_state=42)
        # #train logistic regression model
        clf = LogisticRegression(random_state=0,max_iter=400).fit(X_train, y_train)

        print(f"accuracy of hidden state level {v} is {clf.score(X_test, y_test)}")
        #log accuracy of each hidden state level to wandb
        wandb.log({f"accuracy of hidden state level for dataset of same size{v}":clf.score(X_test, y_test)})
for epoch in range(1, config.epochs + 1):
    print(f"Epoch {epoch}")
    n_correct = 0
    n_samples = 0
    for batch_iter,batch in enumerate(train_dataloader):
        best_model.train()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = best_model(batch["input_values"])
        loss = criterion(outputs["logits"], batch["labels"]) 
        print(f"loss in batch {batch_iter+1} is {loss}")
        #log loss using wandb
        wandb.log({"loss":loss})
        with torch.no_grad():
            eval_hidden_func(eval_dataloader_hidden, best_model)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    with torch.no_grad():
        eval_func(eval_dataloader, best_model)
    
       
    


        
    
breakpoint()
print(trainer.train())
trainer.save_model( f"/pretrained/{model_name}_bestmodel_only_calssifier")
print(trainer.evaluate())

