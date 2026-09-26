# %% [06] Predict: 3 validation images, class names, softmax probabilities, top-4
X_new, y_new = next(iter(val_loader))
X_new, y_new = X_new[:3].to(device), y_new[:3].to(device)

model.eval()
with torch.no_grad():
    y_logits = model(X_new)

y_pred = y_logits.argmax(dim=1)
print("Predicted:", [class_names[i] for i in y_pred])
print("True:     ", [class_names[i] for i in y_new])
print("Correct:  ", (y_pred == y_new).tolist())

y_proba = F.softmax(y_logits, dim=1)
print("\nSoftmax probabilities (rounded):")
print(y_proba.round(decimals=3).cpu())

top_k = torch.topk(y_proba, k=4, dim=1)
for i, (probas, indices) in enumerate(zip(top_k.values, top_k.indices)):
    top4 = ", ".join(f"{class_names[idx]} {p:.1%}" for p, idx in zip(probas, indices))
    print(f"Image {i} (true: {class_names[y_new[i]]}) top-4: {top4}")
