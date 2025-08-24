import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# 다른 모듈에서 필요한 함수 및 클래스 임포트
from utils import set_seed, plot_history, plot_confmat
from data_loader import get_CIFAR10
from model import ResNet18
from trainer import train, evaluate

def main():
    # --- 1. 기본 설정 및 하이퍼파라미터 ---
    set_seed(42)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"DEVICE INFO: {DEVICE}")

    # 고정된 최적 하이퍼파라미터
    NUM_EPOCHS = 300
    PATIENCE = 25
    BATCH_SIZE = 128  # 배치 사이즈는 일반적으로 64, 128 등 2의 거듭제곱을 사용합니다.
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    BEST_MODEL_PATH = 'best_model.pth'

    # --- 2. 데이터 로더 준비 ---
    train_loader, val_loader, test_loader = get_CIFAR10(train_ratio=0.9, batch_size=BATCH_SIZE)

    # --- 3. 모델, 손실 함수, 옵티마이저, 스케줄러 정의 ---
    model = ResNet18().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    # AdamW 옵티마이저와 CosineAnnealingLR 스케줄러 사용
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # --- 4. 모델 학습 ---
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print("===== Start Training =====")
    for epoch in tqdm(range(1, NUM_EPOCHS + 1), desc="Training Progress"):
        # Train
        train_results = train(model, optimizer, criterion, train_loader, DEVICE)
        history['train_loss'].append(train_results['loss'])
        history['train_acc'].append(train_results['accuracy'])

        # Validate
        val_results = evaluate(model, criterion, val_loader, DEVICE)
        history['val_loss'].append(val_results['loss'])
        history['val_acc'].append(val_results['accuracy'])
        
        # 스케줄러 업데이트
        scheduler.step()

        # tqdm 진행률 표시줄에 현재 loss, acc 표시
        tqdm.write(f"[Epoch {epoch:03d}] Train Loss: {train_results['loss']:.4f}, Val Loss: {val_results['loss']:.4f}, Val Acc: {val_results['accuracy']:.2f}%")

        # Early Stopping 및 Best Model 저장
        if val_results['loss'] < best_val_loss:
            best_val_loss = val_results['loss']
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            tqdm.write(f"✨ New Best Model Saved! Val Loss: {best_val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    print("===== Training Finished =====")
    plot_history(history['train_loss'], history['val_loss'], history['train_acc'], history['val_acc'])


    # --- 5. 최종 테스트 평가 ---
    if os.path.exists(BEST_MODEL_PATH):
        print("\n===== Evaluating on Test Set with the Best Model =====")
        # 가장 성능이 좋았던 모델 state를 로드
        final_model = ResNet18().to(DEVICE)
        final_model.load_state_dict(torch.load(BEST_MODEL_PATH))

        final_test_results = evaluate(final_model, criterion, test_loader, DEVICE, is_test=True)

        print(f"Final Test Loss: {final_test_results['loss']:.4f}, Final Test Accuracy: {final_test_results['accuracy']:.2f}%")

        plot_confmat(final_test_results['conf_matrix'], "Final Best Model", "final_best_model_confusion_matrix.png")
        print("Saved the final confusion matrix.")
    else:
        print("\nNo best model was saved. Final evaluation skipped.")


if __name__ == '__main__':
    main()