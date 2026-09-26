# %% [05] Plot: training and validation accuracy per epoch
epochs = range(1, len(history["train_metrics"]) + 1)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs, history["train_metrics"], marker="o", label="Training accuracy")
ax.plot(epochs, history["valid_metrics"], marker="s", label="Validation accuracy")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.set_title("FashionMNIST MLP: accuracy per epoch")
ax.set_xticks(list(epochs))
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()
