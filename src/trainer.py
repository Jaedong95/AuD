import os 
import numpy as np 
import random
import pickle
import torch
from torch import nn
from transformers import BertConfig, BertTokenizer, BertModel
from transformers import AdamW, get_linear_schedule_with_warmup

def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

class BertRegressor(nn.Module):
    def __init__(self, config, model):
        super(BertRegressor, self).__init__()
        self.model = model
        self.linear = nn.Linear(config.hidden_size, 128)
        self.relu = nn.ReLU()
        self.out = nn.Linear(128, 1)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = outputs.last_hidden_state[:, 0, :]
        # print(f'logits: {len(logits)}, {len(logits[0])}')
        x = self.linear(logits)
        x = self.relu(x)
        score = self.out(x)
        # print(f'score: {score}')
        return score 

class BertRegTrainer():
    def __init__(self, config, training_config, model, train_dataloader, eval_dataloader):
        self.config = config
        self.training_config = training_config
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
    def set_seed(self):
        random.seed(self.training_config.seed)
        np.random.seed(self.training_config.seed)
        torch.manual_seed(self.training_config.seed)
        if not self.training_config.no_cuda and torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.training_config.seed)
    
    def train(self):
        nb_eval_steps = 0
        train_rmse = []; eval_rmse = []
        t_total = len(self.train_dataloader) // self.training_config.gradient_accumulation_steps * self.training_config.num_epochs

        optimizer = AdamW(self.model.parameters(), lr=self.training_config.learning_rate, eps=self.training_config.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * self.training_config.warmup_proportion), \
                                                    num_training_steps=t_total)
        
        criterion = RMSELoss
        best_loss = 9999 
        
        self.model.zero_grad()
        for epoch in range(int(self.training_config.num_epochs)):
            train_loss = 0.0; eval_loss = 0.0 
            
            for _ , batch in enumerate(self.train_dataloader):
                self.model.train()
                batch = tuple(t.to(self.training_config.device) for t in batch)
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }
                outputs = self.model(**inputs)
                # print(f'output: {type(outputs)}, {outputs.squeeze}')
                loss = criterion(outputs.squeeze(), batch[3].type_as(outputs))   # batch[3]: label
                loss.backward()
                
                train_loss += loss.item()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                
                self.model.zero_grad()
            
            print(f'epoch: {epoch + 1} done, train_loss: {train_loss / len(self.train_dataloader)}')
            train_rmse.append(train_loss / len(self.train_dataloader))

            for _ , batch2 in enumerate(self.eval_dataloader):
                self.model.eval()
                batch2 = tuple(t.to(self.training_config.device) for t in batch2)

                with torch.no_grad():
                    inputs = {
                        "input_ids": batch2[0],
                        "attention_mask": batch2[1],
                        "token_type_ids": batch2[2],
                    }
                    label2 = batch2[3]
                    outputs = self.model(**inputs)
                    tmp_eval_loss = criterion(outputs.squeeze(), label2.type_as(outputs))
                    eval_loss += tmp_eval_loss.mean().item()
                    
                nb_eval_steps += 1

            eval_loss = eval_loss / nb_eval_steps
            eval_rmse.append(eval_loss)
            if eval_loss < best_loss:
                best_loss = eval_loss
                es = 0
                print(f'save best loss state model & log(epoch {epoch + 1})')
                self.save_model(os.path.join(self.training_config.model_path, f'bert_bws_{epoch}.pt'))
            else:
                es += 1
                print("Counter {} of 5".format(es))

            if es > 4:
                print("Early stopping with best_loss: ", best_loss, "and val_loss for this epoch: ", eval_loss, "...")
                break

        self.save_log(train_rmse, eval_rmse, epoch+1)
        return train_rmse, eval_rmse
            
    def save_log(self, train_mse, eval_mse, epoch):
        with open(os.path.join(self.training_config.log_path, f'train_{epoch}_mse.pickle'), 'wb') as f:
            pickle.dump(train_mse, f, pickle.HIGHEST_PROTOCOL)  
        
        with open(os.path.join(self.training_config.log_path, f'eval_{epoch}_mse.pickle'), 'wb') as f:
            pickle.dump(eval_mse, f, pickle.HIGHEST_PROTOCOL)  
    
    def save_model(self, model_name):
        torch.save(self.model.state_dict(), model_name)


class BertClsTrainer():
    def __init__(self, config, training_config, model, train_dataloader, eval_dataloader):
        self.config = config
        self.training_config = training_config
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
    def set_seed(self):
        random.seed(self.training_config.seed)
        np.random.seed(self.training_config.seed)
        torch.manual_seed(self.training_config.seed)
        if not self.training_config.no_cuda and torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.training_config.seed)
    
    def train(self):
        train_acc_list = []; eval_acc_list = [] 
        train_loss_list = []; eval_loss_list = []
        best_loss = 9999; 
        t_total = len(self.train_dataloader) // self.training_config.gradient_accumulation_steps * self.training_config.num_epochs

        optimizer = AdamW(self.model.parameters(), lr=self.training_config.learning_rate, eps=self.training_config.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * self.training_config.warmup_proportion), \
                                                    num_training_steps=t_total)

        self.model.zero_grad()
        for epoch in range(int(self.training_config.num_epochs)):
            train_acc = 0.0; eval_acc = 0.0
            train_loss = 0.0; eval_loss = 0.0 

            for step, batch in enumerate(self.train_dataloader):
                self.model.train()
                batch = tuple(t.to(self.training_config.device) for t in batch)
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels": batch[3]
                }
                outputs = self.model(**inputs)
                criterion = nn.CrossEntropyLoss()
                # loss = outputs[0]
                # y_pred = torch.max(outputs[1], 1)[1]
                y_pred = outputs[1]
                y_true = batch[3]
                loss = criterion(y_pred, y_true)
                # print(outputs[0], outputs[1], batch[3])
                loss.backward()
                train_loss += loss.item()
                train_acc += self.calc_accuracy(outputs[1], batch[3])

                optimizer.step()
                scheduler.step()
                self.model.zero_grad()
        
            train_acc = train_acc / (step + 1)
            train_loss = train_loss / (step + 1)
            print(f'epoch: {epoch}, train_loss: {train_loss}')
            train_acc_list.append(train_acc)
            train_loss_list.append(train_loss)

            for step2, batch2 in enumerate(self.eval_dataloader):
                self.model.eval()
                batch2 = tuple(t.to(self.training_config.device) for t in batch2)

                with torch.no_grad():
                    inputs = {
                        "input_ids": batch2[0],
                        "attention_mask": batch2[1],
                        "token_type_ids": batch2[2],
                        "labels": batch2[3]
                    }
                    outputs = self.model(**inputs)
                    _ , logits = outputs[:2]
                    loss2 = criterion(logits, batch2[3])
                    eval_loss += loss2.item()              
                    # eval_loss += tmp_eval_loss.mean().item()
                    eval_acc += self.calc_accuracy(outputs[1], batch2[3]) 
            eval_loss = eval_loss / (step2 + 1)
            eval_acc = eval_acc / (step2 + 1)
            eval_acc_list.append(eval_acc)
            eval_loss_list.append(eval_loss)
            print(f'epoch: {epoch}, eval_loss: {eval_loss}')
            
            if eval_loss < best_loss:
                best_loss = eval_loss
                es = 0
                print(f'save best loss state model & log(epoch {epoch + 1})')
                self.save_model(os.path.join(self.training_config.model_path, f'bert_dsm_{epoch}.pt'))
            else:
                es += 1
                print("Counter {} of 5".format(es))

            if es > 4:
                print("Early stopping with best_loss: ", best_loss, "and val_loss for this epoch: ", eval_loss, "...")
                break

        self.save_log(train_loss_list, eval_loss_list, epoch)
        return train_loss_list, eval_loss_list

    def calc_accuracy(self, X,Y):
        _ , max_indices = torch.max(X, 1)
        train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
        return train_acc
    
    def compute_metrics(self, labels, preds):
        assert len(preds) == len(labels)
        acc = (labels == preds).mean()
        return {"acc": acc}

    def save_log(self, train_loss, eval_loss, epoch):
        with open(os.path.join(self.training_config.log_path, f'train_{epoch}_loss.pickle'), 'wb') as f:
            pickle.dump(train_loss, f, pickle.HIGHEST_PROTOCOL)  
        
        with open(os.path.join(self.training_config.log_path, f'eval_{epoch}_loss.pickle'), 'wb') as f:
            pickle.dump(eval_loss, f, pickle.HIGHEST_PROTOCOL)  
    
    def save_model(self, model_name):
        torch.save(self.model.state_dict(), model_name)