import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_mnist_data(batch_size=128):
    
    # We normalize the pixel values to [-1,-1] (not [0,1]).
    # This is beacuse the generators final layer uses tanh. This is
    # the standard convention for GANs, since Tanh trains more stalbly than sigmoid
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # We only need train data
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    print(f"Train samples length : {len(train_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    
    return train_loader