model.eval()
X_new, y_new = next(iter(valid_loader))
X_new, y_new = X_new[:3].to(device), y_new[:3]
with torch.no_grad():
    y_pred_logits = model(X_new)
y_pred = y_pred_logits.argmax(dim=1).cpu()
print("Predicted:", [class_names[i] for i in y_pred])
print("True:     ", [class_names[i] for i in y_new])

y_proba = F.softmax(y_pred_logits, dim=1).cpu()
print("Probabilities:\n", y_proba.round(decimals=3))

y_top4_logits, y_top4_indices = torch.topk(y_pred_logits, k=4, dim=1)
y_top4_probas = F.softmax(y_pred_logits, dim=1).gather(1, y_top4_indices).cpu()
for i in range(3):
    top4 = ", ".join(f"{class_names[j]} ({p:.3f})"
                     for j, p in zip(y_top4_indices[i].cpu(), y_top4_probas[i]))
    print(f"Image {i}: top-4 = {top4}")
