# %% [09] Hyperparameter tuning: lr and n_hidden with Optuna
import optuna

N_TRIALS = 5
N_TRIAL_EPOCHS = 10


def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    n_hidden = trial.suggest_int("n_hidden", 20, 300)

    torch.manual_seed(SEED)
    trial_model = ImageClassifier(n_hidden1=n_hidden, n_hidden2=n_hidden, n_classes=len(class_names)).to(device)
    trial_optimizer = torch.optim.SGD(trial_model.parameters(), lr=lr)
    trial_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=len(class_names)).to(device)

    trial_history = train2(
        trial_model, trial_optimizer, criterion, trial_accuracy, train_loader, val_loader, N_TRIAL_EPOCHS
    )
    return trial_history["valid_metrics"][-1]


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=N_TRIALS)

print("\nAll trials:")
for t in study.trials:
    print(f"  Trial {t.number}: lr={t.params['lr']:.2e}, n_hidden={t.params['n_hidden']}, valid accuracy={t.value:.2%}")
print(f"Best params: {study.best_params}")
print(f"Best valid accuracy: {study.best_value:.2%}")
