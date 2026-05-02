import torch, json

# See the best architecture
checkpoint = torch.load('best_checkpoint.pt')
print("Best val_acc:", checkpoint['best_val_acc'])
print("Best architecture:")
for i, layer in enumerate(checkpoint['best_architecture']):
    print(f"  Layer {i}: {layer}")