model_data = {
    "model_state_dict": model.state_dict(),
    "model_hyperparameters": {"n_inputs": 28 * 28, "n_hidden1": 300,
                              "n_hidden2": 100, "n_classes": 10},
}
torch.save(model_data, "my_fashion_mnist.pt")

loaded_data = torch.load("my_fashion_mnist.pt", weights_only=True)
new_model = ImageClassifier(**loaded_data["model_hyperparameters"]).to(device)
new_model.load_state_dict(loaded_data["model_state_dict"])
new_model.eval()

with torch.no_grad():
    same = torch.allclose(model(X_new), new_model(X_new))
print("Reloaded model gives identical predictions:", same)
