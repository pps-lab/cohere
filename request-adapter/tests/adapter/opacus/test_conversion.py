import pytest
import torch
import math

from opacus.validators import ModuleValidator
from torchvision import models
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10


from opacus import PrivacyEngine

from request_adapter.adapter.opacus.converter import convert_to_autodp


@pytest.fixture
def model():
    device = torch.device("cpu")
    model = models.resnet18(num_classes=10)
    model = ModuleValidator.fix(model)
    ModuleValidator.validate(model, strict=False)
    model = model.to(device)
    return model

@pytest.fixture
def optimizer(model):

    def get_optimizer(model):
        LR = 1e-3
        return torch.optim.RMSprop(model.parameters(), lr=LR)

    return get_optimizer

@pytest.fixture
def train_loader():

    BATCH_SIZE = 512

    # These values, specific to the CIFAR10 dataset, are assumed to be known.
    # If necessary, they can be computed with modest privacy budgets.
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV),
    ])

    DATA_ROOT = '../data/cifar10'
    train_dataset = CIFAR10(root=DATA_ROOT, train=True, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
    )

    return train_loader


def test_opacus_calibration(model, optimizer, train_loader):

    TARGET_EPSILON = 3.0
    TARGET_DELTA = 1e-8

    EPOCHS = 20
    MAX_GRAD_NORM = 2.5

    privacy_engine = PrivacyEngine(accountant="rdp")

    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer(model),
        data_loader=train_loader,
        epochs=EPOCHS,
        target_epsilon=TARGET_EPSILON,
        target_delta=TARGET_DELTA,
        max_grad_norm=MAX_GRAD_NORM,
    )

    n_batches = len(train_loader)

    mech_autodp, _mech_autodp_amplified, _mechanisms_info = convert_to_autodp(optimizer=optimizer, n_batches=n_batches, epochs=EPOCHS, population_prob=1.0)

    epsilon_autodp = mech_autodp.get_approxDP(TARGET_DELTA)

    #DEFAULT_ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

    #def to_adp_tight(eps_rdp, alpha, delta):
    #    return eps_rdp + math.log((alpha - 1) / alpha) - (math.log(delta) + math.log(alpha)) / (alpha - 1)

    #min_epsilon_found = min(to_adp_tight(mech_autodp.get_RDP(alpha), alpha, TARGET_DELTA) for alpha in DEFAULT_ALPHAS)
    #print(f"MIN_EPSILON_FOUND_TIGHT={min_epsilon_found}")
    assert math.isclose(TARGET_EPSILON, epsilon_autodp, abs_tol=0.01), f"The accounting in autodp does not match the accounting in opacus: autodp-eps={epsilon_autodp}   opacus-eps={TARGET_EPSILON}"