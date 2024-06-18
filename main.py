import argparse
import math
import pickle
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from seqeval.metrics import f1_score as seq_f1_score
from seqeval.scheme import IOB2
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling,TrainerCallback
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
import traceback 
import numpy as np
import torch.optim as optim



class CustomTextDataset(Dataset):
    def __init__(self, tokenizer, texts, max_length, sample_size=5000):
        texts = texts[:sample_size]
        self.tokenizer = tokenizer
        self.inputs = tokenizer(texts, truncation=True, padding='max_length', max_length=max_length, return_tensors="pt")

    def __len__(self):
        return len(self.inputs.input_ids)

    def __getitem__(self, idx):
        return {key: self.inputs[key][idx] for key in self.inputs.keys()}


class JointTrainingDataset(Dataset):
    def __init__(self, df, vocab,  rel2idx):
        self.df = df
        self.vocab = vocab
        self.rel2idx = rel2idx
        self.unkidx = vocab['<unk>']

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = [self.vocab.get(token, self.unkidx) for token in row['utterances']]
        rels = row['Core Relations']
        return torch.Tensor(text).long(), torch.Tensor(rels).long()

    def __len__(self):
        return len(self.df)

def pretrain_wikitext():
    try:
        print("Pretraining on WikiText...")
        wikitext_dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        max_length = 128
        train_dataset = CustomTextDataset(tokenizer, wikitext_dataset['train']['text'], max_length)
        validation_dataset = CustomTextDataset(tokenizer, wikitext_dataset['validation']['text'], max_length)

        model = GPT2LMHeadModel.from_pretrained("gpt2")

        training_args = TrainingArguments(
            output_dir="./gpt2-finetuned-wikitext",
            overwrite_output_dir=True,
            num_train_epochs=1,
            per_device_train_batch_size=4,
            logging_steps=50,
            save_steps=1000,
            evaluation_strategy="steps",
            eval_steps=1000,
        )

        class PlotLosses(TrainerCallback):
            def __init__(self):
                self.losses = []

            def on_log(self, args, state, control, logs=None, **kwargs):
                if 'loss' in logs:
                    self.losses.append(logs['loss'])

        plot_losses = PlotLosses()

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            data_collator=data_collator,
            callbacks=[plot_losses],
        )

        trainer.train()
        eval_results = trainer.evaluate()
        print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

        plt.figure(figsize=(10, 6))
        plt.plot(plot_losses.losses, label="Training Loss")
        plt.xlabel("Logging Steps")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Time")
        plt.legend()
        #plt.show()
        plt.savefig("pretaining_losses.png")

        return model, tokenizer

    except Exception as e:
        print(f"Error during pretraining: {e}")
        return None, None


def finetune_custom(model, tokenizer):
    try:
        print("Fine-tuning on custom dataset...")
        df = pd.read_csv("hw1_train.csv")

        vocab = {'<pad>': 0, '<unk>': 1}
        for idx, token in enumerate(df['utterances'].explode().unique(), start=2):
            vocab[token] = idx

        def tokenize(s):
            return s.split()
        
        def collate(batch):
            texts, rels = zip(*batch)
            texts = pad_sequence(texts, batch_first=True, padding_value=vocab['<pad>'])
            rels = torch.stack(rels, dim=0)
            pad_mask = (texts != vocab['<pad>'])
            return texts, rels, pad_mask

        df['utterances'] = df['utterances'].apply(tokenize)
        df['IOB Slot tags'] = df['IOB Slot tags'].apply(lambda x:x.replace('_','_')).apply(tokenize)
        df['Core Relations']=df['Core Relations'].fillna("").apply(tokenize)
        idx2rel = dict(enumerate(df["Core Relations"].explode().dropna().unique()))
        rel2idx = {v: k for k, v in idx2rel.items()}
        rel2idx = {v:k for k,v in idx2rel.items()}
        mlb =MultiLabelBinarizer(classes = list(rel2idx.keys()))
        mlb.fit(df['Core Relations'])
        df['Core Relations'] = mlb.transform(df['Core Relations']).tolist()
         

        with open('mlb_model.pkl', 'wb') as file:
          pickle.dump(mlb, file)
        train_df, val_df = train_test_split(df, test_size=0.1)
        train_ds = JointTrainingDataset(train_df, vocab, rel2idx)
        val_ds = JointTrainingDataset(val_df, vocab, rel2idx)
        train_dataloader = DataLoader(train_ds, batch_size=4, collate_fn=collate)
        val_dataloader = DataLoader(val_ds, batch_size=4, collate_fn=collate)

        rel_loss = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
        model.config.pad_token_id = 0
        model.train()

        epochs = 10  # Number of epochs for fine-tuning

        losses = []
        val_f1_scores = []

        for epoch in range(epochs):
            epoch_losses = []

            for idx, (texts, rels, pad_mask) in enumerate(tqdm(train_dataloader)):
                optimizer.zero_grad()
                rel_logits = model(texts, attention_mask=pad_mask)[0]  # Modify to use attention mask
                loss = rel_loss(rel_logits, rels.float())
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

            losses.extend(epoch_losses)
            mean_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch+1} Loss: {mean_loss}")

            # Validation
            model.eval()
            all_rel_preds = []
            all_rels = []

            for idx, (texts, rels, pad_mask) in enumerate(tqdm(val_dataloader)):
                rel_logits = model(texts, attention_mask=pad_mask)[0]  # Modify to use attention mask
                rel_preds = torch.sigmoid(rel_logits) > 0.5
                all_rels.extend(rels)
                all_rel_preds.extend(rel_preds)

            val_f1 = f1_score(all_rels, all_rel_preds, average="micro")
            val_f1_scores.append(val_f1)
            print("Validation rel score:", val_f1)

        # Plot and save losses and F1 scores
        plt.figure(figsize=(10, 6))

        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(losses, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Epochs")
        plt.legend()

        # Plot F1 scores
        plt.subplot(1, 2, 2)
        plt.plot(val_f1_scores, label="Validation F1 Score", color='r')
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.title("Validation F1 Score Over Epochs")
        plt.legend()

        plt.tight_layout()
        plt.savefig("fine_tuning_metrics.png")  # Save the plot as PNG
    except Exception as e:
        print(f"Error during fine-tuning: {e}")
        traceback.print_exc() 

class PredictionDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

def generate_predictions(model_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer.pad_token = tokenizer.eos_token
    # Load your test dataset
    df_test = pd.read_csv("hw1_test.csv")
    texts = df_test['utterances'].tolist()

    # Correctly load the MultiLabelBinarizer instance
    with open('mlb_model.pkl', 'rb') as file:  # Use 'rb' to read in binary mode
        mlb = pickle.load(file)

    # Tokenize the test data
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    test_dataset = PredictionDataset(encodings)

    # Use Trainer for predictions
    training_args = TrainingArguments(
        output_dir='/tmp/test_predictions',
        per_device_eval_batch_size=64,
        do_predict=True
    )
    trainer = Trainer(
        model=model,
        args=training_args
    )

    # Generate predictions
    predictions = trainer.predict(test_dataset)
    preds = torch.sigmoid(torch.from_numpy(predictions.predictions)).numpy() > 0.5

    # Decode predictions to labels
    decoded_preds = mlb.inverse_transform(preds)

    # Save the decoded predictions
    df_test['predicted_relations'] = ["; ".join(pred) if pred else "No relation" for pred in decoded_preds]
    df_test.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")




def main(train, test, model_path, save_model, output):
    try:
        # Initialize the tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        if train:
            # Pretrain on WikiText
            model, tokenizer = pretrain_wikitext()

            if save_model:
                model.save_pretrained(os.path.join(model_path, "pretrained_model"))
                print("Pretained model saved to" + os.path.join(model_path, "pretrained_model"))
            
            # Fine-tune on custom dataset
            model = AutoModelForSequenceClassification.from_pretrained(os.path.join(model_path, "pretrained_model"), num_labels=18)
            finetune_custom(model, tokenizer)
            
            if save_model:
                model.save_pretrained(os.path.join(model_path, "finetuned_model"))
                print(f"Fine-tuned model saved to" + os.path.join(model_path, "finetuned_model"))
        
        if test:
            # Generate predictions using the saved model
            generate_predictions(model_path, output)
    
    except Exception as e:
        print(f"Error during main execution: {e}")
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="Run both pretraining and fine-tuning.", action="store_true")
    parser.add_argument("--test", help="Generate predictions using the saved model.", action="store_true")
    parser.add_argument("--model_path", help="Path to load/save the model.", type=str, default="./gpt2-finetuned")
    parser.add_argument("--save_model", help="Save the fine-tuned model.", action="store_true")
    parser.add_argument("--output", help="Path to save the predictions.", type=str, default="./predictions.txt")
    args = parser.parse_args()

    main(args.train, args.test, args.model_path, args.save_model, args.output)
