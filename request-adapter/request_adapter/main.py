
from dataclasses import asdict
from datetime import timedelta

import pandas as pd
import torch
import torchvision.transforms as transforms
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from pyspark.sql import SparkSession
from tmlt.analytics.keyset import KeySet
from tmlt.analytics.privacy_budget import RhoZCDPBudget
from tmlt.analytics.query_builder import QueryBuilder
from tmlt.analytics.session import Session
from torchvision import models
from torchvision.datasets import CIFAR10
from request_adapter.adapter.libconverter import (ConverterConfig,
                                                     create_opacus_request,
                                                     create_tumult_request)


def demo_tumult():
    # 1. SETUP
    df = pd.DataFrame([["0", 1, 0], ["0", 1, 0], ["1", 0, 1]], columns=["A", "B", "X"])

    spark = SparkSession.builder.getOrCreate()
    private_data = spark.createDataFrame(df)

    session = Session.from_dataframe(
        privacy_budget=RhoZCDPBudget(float('inf')), # session budget is not used
        source_id="data",
        dataframe=private_data,
    )

    # 2. BUILDING THE TUMULT QUERY
    query_expr = QueryBuilder("data").groupby(KeySet.from_dict({"A": ["0", "1"]})).average("B", 0.0, 5.0)


    # 3. CONVERTING THE QUERY TO A REQUEST
    config = ConverterConfig(active_time_window=timedelta(weeks=12), allocation_interval=timedelta(weeks=1))
    request = create_tumult_request(session=session, query_expr=query_expr, budget=RhoZCDPBudget(0.2), converter_config=config, population_dnf=None, population_prob=1.0, utility=5)

    print("\n\n\nTUMULT DEMO REQUEST: ")
    print(asdict(request))
    print("=========================================")



def demo_opacus():

    device = torch.device("cpu")
    model = models.resnet18(num_classes=10)
    model = ModuleValidator.fix(model)
    ModuleValidator.validate(model, strict=False)
    model = model.to(device)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)


    train_loader = _train_loader()

    privacy_engine = PrivacyEngine(accountant="rdp")

    EPOCHS = 20

    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=EPOCHS,
        target_epsilon=3.0,
        target_delta=1e-8,
        max_grad_norm=2.5,
    )

    config = ConverterConfig(active_time_window=timedelta(weeks=12), allocation_interval=timedelta(weeks=1))
    request = create_opacus_request(optimizer=optimizer, n_batches_per_epoch=len(train_loader), epochs=EPOCHS, converter_config=config, population_dnf=None, population_prob=1.0, utility=5)

    print("\n\n\nOPACUS DEMO REQUEST: ")
    print(asdict(request))
    print("=========================================")


def _train_loader():
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



if __name__ == "__main__":
    demo_tumult()
    demo_opacus()