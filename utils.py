import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def set_seed(seed=42):
    """재현성을 위한 시드 설정 함수"""
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_history(train_loss_list, val_loss_list, train_acc_list, val_acc_list, lr, batch_size):
    """학습 과정의 Loss와 Accuracy를 시각화하고 저장하는 함수"""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list, label='Train Loss')
    plt.plot(val_loss_list, label='Validation Loss')
    plt.title(f"Loss (LR: {lr}, Batch: {batch_size})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_list, label='Train Accuracy')
    plt.plot(val_acc_list, label='Validation Accuracy')
    plt.title(f"Accuracy (LR: {lr}, Batch: {batch_size})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # 'plots' 폴더가 없으면 생성
    if not os.path.exists('plots'):
        os.makedirs('plots')
        
    filename = f"plots/history_lr{lr}_batch{batch_size}.png"
    plt.savefig(filename)
    plt.close()

def plot_confmat(conf_matrix, title_suffix, filename):
    """Confusion Matrix를 시각화하고 저장하는 함수"""
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    plt.figure(figsize=(10, 8)) # plt.Figure -> plt.figure로 수정
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=cifar10_classes, yticklabels=cifar10_classes)

    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f"Confusion Matrix ({title_suffix})")

    # 'plots' 폴더가 없으면 생성
    if not os.path.exists('plots'):
        os.makedirs('plots')

    plt.savefig(filename)
    plt.close()
