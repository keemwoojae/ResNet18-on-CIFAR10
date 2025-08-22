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
    # --- 기본 설정 ---
    set_seed(42)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"DEVICE INFO: {DEVICE}")

    # --- 하이퍼파라미터 설정 ---
    NUM_EPOCHS = 300
    PATIENCE = 25
    learning_rates = [0.1, 0.01]
    batch_sizes = [64, 128]

    results = []
    best_overall_loss = float('inf')
    best_hyperparams = {}

    search_num = 1
    total_searches = len(learning_rates) * len(batch_sizes)
    with tqdm(total=total_searches, desc='Hyperparameters Search') as outer_pbar:
        for lr in learning_rates:
            for batch_size in batch_sizes:
                outer_pbar.set_description(f"Search #{search_num}: LR={lr}, Batch={batch_size}")
                
                train_loader, val_loader, _ = get_CIFAR10(train_ratio=0.9, batch_size=batch_size)

                model = ResNet18().to(DEVICE)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1)

                best_run_loss = float('inf'); best_run_acc = 0; patience_counter = 0
                train_loss_list, val_loss_list = [], []
                train_acc_list, val_acc_list = [], []

                epoch_pbar = tqdm(range(1, NUM_EPOCHS + 1), desc="Training")
                final_epoch = 0
                for epoch in epoch_pbar:
                    final_epoch = epoch

                    train_results = train(model, optimizer, criterion, train_loader)
                    train_loss = train_results['loss']
                    train_acc = train_results['accuracy']

                    val_results = evaluate(model, criterion, val_loader)
                    val_loss = val_results['loss']
                    val_acc = val_results['accuracy']

                    scheduler.step()

                    train_loss_list.append(train_loss); val_loss_list.append(val_loss)
                    train_acc_list.append(train_acc); val_acc_list.append(val_acc)

                    epoch_pbar.set_postfix({"Train Loss": f"{train_loss:.4f}", "Val Loss": f"{val_loss:.4f}", "Val Acc": f"{val_acc}"})

                    if val_loss < best_run_loss:
                        best_run_loss = val_loss
                        best_run_acc = val_acc
                        patience_counter = 0
                        torch.save(model.state_dict(), f"best_model_run_{search_num}.pth")
                        if val_loss < best_overall_loss:
                            tqdm.write(f"\t✨ New BEST Overall Model! Val Loss: {val_loss:.4f} (Search #{search_num})")
                            best_overall_loss = val_loss
                            best_hyperparams = {'lr': lr, 'batch_size': batch_size}
                            torch.save(model.state_dict(), 'best_overall_model.pth')
                    else:
                        patience_counter += 1
                        if patience_counter >= PATIENCE:
                            tqdm.write(f"Early stopping at epoch {epoch} for Search #{search_num}")
                            break

                results.append({'search_num': search_num,
                                'learning_rate': lr,
                                'batch_size': batch_size,
                                'best_val_loss': best_run_loss,
                                'best_val_acc': best_run_acc,
                                'total_epochs': final_epoch})
                
                if os.path.exists(f"best_model_run_{search_num}.pth"):
                    plot_history(train_loss_list, val_loss_list, train_acc_list, val_acc_list)
                
                outer_pbar.update(1)
                search_num += 1

    # --- 모든 탐색 종료 후 최종 결과 정리 ---
    print("\n\n===== All Experiments Finished =====")
    results_df = pd.DataFrame(results).sort_values(by='best_val_loss', ascending=True).reset_index(drop=True)
    print("--- Hyperparameter Search Results ---\n", results_df)
    results_df.to_csv('hyperparameter_search_results_tqdm.csv', index=False)

    # --- 최종 테스트 평가 ---
    if best_hyperparams:
        print(f"\nBest validation loss: {best_overall_loss:.4f} with {best_hyperparams}")
        print("\n==> Evaluating on test set with the best model...")

        final_model = ResNet18().to(DEVICE)
        final_model.load_state_dict(torch.load('best_overall_model.pth'))
        
        _, _, test_loader = get_CIFAR10(train_ratio=0.9, batch_size=best_hyperparams['batch_size'])
        
        final_test_results = evaluate(final_model, criterion, test_loader, is_test=True)
        
        print(f"Final Test Loss: {final_test_results['loss']:.4f}, Final Test Accuracy: {final_test_results['accuracy']:.2f}%")
        
        plot_confmat(final_test_results['conf_matrix'], "Final Best Model", "final_best_model_confusion_matrix.png")
        print("Saved the final confusion matrix.")
    else:
        print("\nNo best model was saved. Final evaluation skipped.")