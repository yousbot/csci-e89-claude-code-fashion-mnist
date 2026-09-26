test_acc = evaluate_tm(model, test_loader, accuracy)
print(f"Test accuracy: {test_acc.item():.4f}")
