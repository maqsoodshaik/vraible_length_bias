from datasets import load_dataset, load_metric,concatenate_datasets,Dataset
from transformers import AutoFeatureExtractor
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer
from torch.utils.data import DataLoader
import torch
import numpy as np
import skimage.measure
#import weights and bias
import wandb
import asyncio
import torch
torch.cuda.empty_cache()
#import Kmeans and pairwise_distances_argmin_min
from sklearn.cluster import KMeans
#import pairwise_distances
from sklearn.metrics.pairwise import pairwise_distances
import pickle
#set seed for reproducibility
#load data from pickle file

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)
batch_size = 64

async def pruning_function(centroids,hidden_states,after_prune_percent,cluster_labels):
    after_prune = []
    for lb,centroid in enumerate(centroids):
        cluster = np.where(cluster_labels == lb)
        distances = pairwise_distances(hidden_states[cluster], centroid.reshape(1, -1))
        threshold = np.percentile(distances, after_prune_percent)
        within_threshold = np.where(distances <= threshold)
        after_prune.extend(within_threshold[0])
    with open(f"after_prune_{after_prune_percent}.pkl", "wb") as f:
        pickle.dump(after_prune, f)
dataset_name = "fleurs"
configs = ['en_us','cmn_hans_cn','fr_fr']
#,'de_de','nl_nl']
labels =["English","Mandarin Chinese","French"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
metric = load_metric("accuracy")
model_checkpoint = "facebook/wav2vec2-base"
label2id, id2label,label2id_int = dict(), dict(),dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label
    label2id_int[label] = i
best_model = AutoModelForAudioClassification.from_pretrained(
    model_checkpoint, num_labels=len(labels),label2id=label2id,
    id2label=id2label,output_hidden_states=True)
best_model =   best_model.to(device)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)

# config.max_duration_en = 7.0
# config.max_duration_cn = 3.0
# config.max_duration_fr = 5.0
max_duration_en = 7.0
max_duration_cn = 3.0
max_duration_fr = 5.0

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
        max_duration = max(max_duration_en,max_duration_cn,max_duration_fr)
        audio_arrays = [x for x in examples["input_values"]]
        inputs = feature_extractor(
            audio_arrays, 
            sampling_rate=feature_extractor.sampling_rate, 
            max_length=int(feature_extractor.sampling_rate * max_duration), 
            padding="max_length" 
        )
        return inputs
train_dataset_list = []
# test_dataset_list = []
for config_lang,duration in zip(configs,[max_duration_en,max_duration_cn,max_duration_fr]):
    dataset_obj = Load_dataset("google/fleurs",config_lang,duration,feature_extractor,label2id_int)
    train_dataset_list.append(dataset_obj.train_split())
    # test_dataset_list.append(dataset_obj.test_split())
dataset_train_combined= concatenate_datasets(
        train_dataset_list
    )

# dataset_test_combined= concatenate_datasets(
#         test_dataset_list
#     )
encoded_dataset_train = dataset_train_combined.map(preprocess_function,batched=True)
# encoded_dataset_test = dataset_test_combined.map(preprocess_function,batched=True)

encoded_dataset_train.set_format("torch")
# encoded_dataset_test.set_format("torch")
# eval_dataloader = DataLoader(encoded_dataset_test, batch_size=16)
train_dataloader = DataLoader(encoded_dataset_train, batch_size=16)
@torch.no_grad()
def eval_hidden_func(eval_dataloader_hidden, best_model,length):
    labels_p= torch.tensor([])
    logits_p = torch.tensor([])
    d = {}
    
    # for x in range(best_model.config.num_hidden_layers+1):
    d[f"hidden_state_12"] = torch.tensor([])
    for batch in eval_dataloader_hidden:
        best_model.eval()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = best_model(batch["input_values"])
        
        # labels_s = batch["labels"].to("cpu")
        # labels_p = torch.cat((labels_p,labels_s),0)
        # logits_p = torch.cat((logits_p,outputs.logits.to("cpu")),0)
        # for num,hidden_state in enumerate(outputs.hidden_states):
                #LogisticRegression from sklearn to predict the language
        hidden_state = outputs.hidden_states[-1].to("cpu")
            
        hidden_state = hidden_state.mean(dim=1)
        hidden_state = hidden_state.reshape(hidden_state.shape[0], -1)
        d[f"hidden_state_12"] = torch.cat((d[f"hidden_state_12"],hidden_state),0)
    # for v in range(best_model.config.num_hidden_layers+1):
    #Do Kmeans clustering on the hidden states
    kmeans = KMeans(n_clusters=len(labels), random_state=0).fit(d[f"hidden_state_12"])
    #get centroids of the clusters
    centroids = kmeans.cluster_centers_
    cluster_labels = kmeans.predict(d[f"hidden_state_12"])  
    after_prune_percentage = (20,40,60,80,90)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.gather(*[pruning_function(centroids,d[f"hidden_state_12"],after_prune_percent,cluster_labels) for after_prune_percent in after_prune_percentage]))
    loop.close()
    
        
    #plot the hidden states in 2 dimension using TSNE with labels_p as labels
    
    # tsne = TSNE(n_components=2, random_state=0)
    # X_2d = tsne.fit_transform(d[f"hidden_state_{v}"])
    # plt.figure(figsize=(10, 10))
    # plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_p)

    #store the plot in wandb as image
    # wandb.log({f"TSNE plot of hidden state {v} of {length}": wandb.Image(plt)})
        
    #plot the logits in 2 dimension using TSNE with labels_p as labels

if __name__ == "__main__":
    with torch.no_grad():
        eval_hidden_func(train_dataloader, best_model,"different")