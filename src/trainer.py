import torch
from tqdm import tqdm
from sklearn.metrics import classification_report
from .utils import LossTracker

class Trainer:
    def __init__(self, model, optimizer, scheduler=None, device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.tracker = LossTracker()

    def train_epoch(self, data_loader):
        self.model.train()
        final_loss = 0
        
        progress_bar = tqdm(data_loader, total=len(data_loader), desc="Training")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            targets = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask)
            
            loss = torch.nn.CrossEntropyLoss()(outputs.logits, targets)
            loss.backward()
            self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            final_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return final_loss / len(data_loader)

    def evaluate(self, data_loader):
        self.model.eval()
        final_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                targets = batch['label'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = torch.nn.CrossEntropyLoss()(outputs.logits, targets)
                final_loss += loss.item()
                
        return final_loss / len(data_loader)

    def predict(self, data_loader):
        self.model.eval()
        predictions = []
        actual_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                actual_labels.extend(batch['label'].numpy())
                
                outputs = self.model(input_ids, attention_mask)
                predictions.extend(torch.argmax(outputs.logits, axis=1).cpu().numpy())
        
        return predictions, actual_labels
