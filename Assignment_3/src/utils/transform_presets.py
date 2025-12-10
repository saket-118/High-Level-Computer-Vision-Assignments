import torchvision.transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# one common eval transform
_eval_cifar10 = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

presets = dict(

    CIFAR10=dict(
        train=T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
        eval=_eval_cifar10,
    ),

    CIFAR10_aug1=dict(
        train=T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
        eval=_eval_cifar10,
    ),

    CIFAR10_aug2=dict(
        train=T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.3, contrast=0.3,
                          saturation=0.3, hue=0.02),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
        eval=_eval_cifar10,
    ),

    CIFAR10_aug3=dict(
        train=T.Compose([
            T.RandAugment(num_ops=2, magnitude=7),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
        eval=_eval_cifar10,
    ),
    CIFAR10_VGG=dict(
        train=T.Compose([
            T.Resize(224),          # Scale shorter side to 224
            T.CenterCrop(224),      # Crop a 224×224 square
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
        eval=T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
    )
)
