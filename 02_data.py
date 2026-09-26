# %% [02] Data: load FashionMNIST, train/val split, DataLoaders
import torchvision.transforms.v2 as T

transform = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])

train_full = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
test_ds = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)
class_names = train_full.classes

train_ds, val_ds = random_split(train_full, [55_000, 5_000], generator=torch.Generator().manual_seed(SEED))

BATCH_SIZE = 32
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

print(f"train: {len(train_ds)} | val: {len(val_ds)} | test: {len(test_ds)}")

image, label = train_ds[0]
print(f"Sample shape: {tuple(image.shape)}, dtype: {image.dtype}, range: [{image.min():.2f}, {image.max():.2f}]")
print(f"Sample label: {label} ({class_names[label]})")
