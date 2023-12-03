import torch
import torch.nn as nn
import torch.optim as optim
import os
import requests
from PIL import Image
from io import BytesIO
from data_utils.data_loader import getDataloader
from evaluate.evaluate import cal_score
from model.model import CNN_Model
from tqdm import tqdm

class Training():
    def __init__(self, config):
        self.num_epochs = config['num_epochs']
        self.learning_rate = config['learning_rate']
        self.momentum = config['momentum']
        self.save_path=config['save_path']
        self.patience = config['patience']
        self.train_loader = getDataloader(config).get_train()
        self.val_loader = getDataloader(config).get_val()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CNN_Model(config).to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr = self.learning_rate, momentum = self.momentum)

    def main(self):
        if not os.path.exists(self.save_path):
          os.makedirs(self.save_path)
        if os.path.exists(os.path.join(self.save_path, 'last_model.pth')):
            checkpoint = torch.load(os.path.join(self.save_path, 'last_model.pth'))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            initial_epoch = checkpoint['epoch'] + 1
            print(f"training from epoch {initial_epoch}")
        else:
            initial_epoch = 0
            print("training for the 1st time...")
            train_loss = 0.
        if os.path.exists(os.path.join(self.save_path, 'best_model.pth')):
            checkpoint = torch.load(os.path.join(self.save_path, 'best_model.pth'))
            best_score = checkpoint['score']
        else:
            best_score = 0.

        print('T R A I N I N G    S T A R T E D')

        for epoch in range(self.num_epochs):
            val_acc = 0.
            val_f1 = 0.
            val_precision = 0.
            val_recall = 0.
            train_loss = 0.

            self.model.train()

            for it,item in enumerate(self.train_loader):
                images, labels = item[0].to(self.device), item[1].to(self.device)
                self.optimizer.zero_grad()
                pred, loss = self.model(images, labels)

                loss.backward()
                self.optimizer.step()
                train_loss += loss

            with torch.no_grad():
                for it,item in enumerate(self.val_loader):
                    images, labels = item[0].to(self.device), item[1].to(self.device)
                    logits = self.model(images)
                    preds = logits.argmax(-1)
                    cm, acc, f1, precision, recall = cal_score(labels.cpu().numpy(),preds.cpu().numpy())
                    val_acc += acc
                    val_f1 += f1
                    val_precision += precision
                    val_recall += recall

            train_loss /= len(self.train_loader)
            val_acc /= len(self.val_loader)
            val_f1 /= len(self.val_loader)
            val_precision /= len(self.val_loader)
            val_recall /= len(self.val_loader)

            print('___________________________________')
            print(f"epoch {epoch + 1}/{self.num_epochs}")
            print(f"train loss: {train_loss:.4f}")
            print(f"val acc: {val_acc:.4f} | val f1: {val_f1:.4f} | val precision: {val_precision:.4f} | val recall: {val_recall:.4f}")
            print('___________________________________')

            score = val_acc

            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'score': score}, os.path.join(self.save_path, 'last_model.pth'))
            
            if epoch > 0 and score <= best_score:
              threshold += 1
            else:
              threshold = 0

            if score > best_score:
                best_score = score
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'score':score}, os.path.join(self.save_path, 'best_model.pth'))
                print(f"saved the best model with accuracy {score:.4f}")

            if threshold >= self.patience:
                print(f"early stopped after epoch {epoch + 1}")
                break

        print(' T R A I N I N G    D O N E !')
