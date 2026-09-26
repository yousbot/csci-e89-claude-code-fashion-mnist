# %% [08] Save/load: state_dict with hyperparameters, verify predictions match
from pathlib import Path

model_hparams = {"n_inputs": 28 * 28, "n_hidden1": 300, "n_hidden2": 100, "n_classes": len(class_names)}
train_hparams = {"lr": LEARNING_RATE, "n_epochs": N_EPOCHS, "batch_size": BATCH_SIZE, "seed": SEED}

checkpoint_path = Path("models/fashion_mnist_mlp.pt")
checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "model_hparams": model_hparams,
        "train_hparams": train_hparams,
        "class_names": class_names,
    },
    checkpoint_path,
)
print(f"Saved checkpoint to {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
loaded_model = ImageClassifier(**checkpoint["model_hparams"]).to(device)
loaded_model.load_state_dict(checkpoint["model_state_dict"])
loaded_model.eval()
print(f"Loaded model with {checkpoint['model_hparams']} (trained with {checkpoint['train_hparams']})")

model.eval()
with torch.no_grad():
    original_logits = model(X_new)
    loaded_logits = loaded_model(X_new)

assert torch.allclose(original_logits, loaded_logits), "Loaded model logits differ from the original"
assert torch.equal(original_logits.argmax(dim=1), loaded_logits.argmax(dim=1))
print("Predictions match:", [checkpoint["class_names"][i] for i in loaded_logits.argmax(dim=1)])
