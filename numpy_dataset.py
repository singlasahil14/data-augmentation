import torch
import torch.utils.data as data
from PIL import Image

class NumpyDataset(data.Dataset):
    """Dataset wrapping data and target numpy arrays.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        img_np (Tensor): contains images.
        target_np (Tensor): contains targets (labels).
    """

    def __init__(self, img_np, target_np, transform=None, target_transform=None):
        assert img_np.shape[0] == target_np.shape[0]
        self.img_tensor = torch.from_numpy(img_np).float()
        self.target_tensor = torch.from_numpy(target_np)
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.img_tensor[index], self.target_tensor[index]
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.img_tensor.size(0)
