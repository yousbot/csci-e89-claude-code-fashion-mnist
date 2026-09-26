# %% [07] Test evaluation: accuracy on the held-out test set
test_accuracy = evaluate_tm(model, test_loader, accuracy).item()
print(f"Validation accuracy (last epoch): {history['valid_metrics'][-1]:.2%}")
print(f"Test accuracy:                   {test_accuracy:.2%}")
