# Request Adapter
<a name="readme-top"></a>

<!-- ABOUT THE PROJECT -->
## About The Project

Cohere exposes an application-layer API where a variety of differentially private systems can express their privacy resource needs.
The prototype [resource planner](./../dp-planner) expects a `--requests` argument containing all candidate requests in a `JSON` file.
Although it is possible to directly express privacy resource requests, employing a request adapter that seamlessly integrates with a DP library significantly streamlines the process.

As part of the prototype, we provide a request adapter that integrates directly with [Tumult Analytics][tumult-url], a DP library designed for aggregate queries on tabular data, as well as [Opacus][opacus-url], a PyTorch-based DP library for ML training.


### Built With

* [![Python][python-shield]][python-url]
* [![Poetry][poetry-shield]][poetry-url]
* [![AutoDP][autodp-shield]][autodp-url]
* [![Tumult Analytics][tumult-shield]][tumult-url]
* [![Opacus][opacus-shield]][opacus-url]



<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

To get a local demo up and running follow these simple steps.

### Prerequisites

* [Python Poetry](https://python-poetry.org/)
    ```sh
    curl -sSL https://install.python-poetry.org | python3 -
    ```

* [Make](https://www.gnu.org/software/make/)


* [Java](https://www.java.com/en/)
    ```sh
    sudo apt install default-jre
    ```

* Local clone of the repository (with submodules)
    ```sh
    git clone --recurse-submodules git@github.com:pps-lab/cohere.git
    ```


### Installation


1. Setup PyTorch (may require adding CUDA to the `LD_LIBRARY_PATH` environment variable):
    ```
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.2/lib64/
    ```


2. Run the demo for Tumult Analytics and Opacus (from the project root):
    ```sh
    make request-adapter-demo
    ```



<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

The demo in `request-adapter/request_adapter/main.py` illustrates how to use the request adapter for [Tumult Analytics][tumult-url] and [Opacus][opacus-url].
In addition, there is a set of tests to ensure that the privacy costs in the requests match the costs of the DP library.


### Tumult Analytics

A [Tumult Analytics][tumult-url] application starts with a `Session`, which lists all available tables and initializes privacy accounting parameters.
Developers then utilize the `QueryBuilder` to express aggregate queries using a SQL-like Domain-Specific Language (DSL).
Normally, these queries would be executed within the `Session`.
However, in the context of Cohere, rather than direct execution, our aim is to formulate a resource request.

To achieve this, we construct a `ConverterConfig` and subsequently invoke `create_tumult_request(..)`.
This function creates a Cohere request, based on inputs such as the `Session`, the `QueryExpr` from the  `QueryBuilder`, and the queries' privacy budget.



```Python

# init a standard Tumult Session
session = Session.from_dataframe(
    privacy_budget=RhoZCDPBudget(float('inf')), # budget is unused
    source_id="data",
    dataframe=private_data,
)

# building a Tumult query
query_expr = QueryBuilder("data")
                .groupby(KeySet.from_dict({"A": ["0", "1"]}))
                .average("B", 0.0, 5.0)

# convert to a Cohere Request
config = ConverterConfig(
                active_time_window=timedelta(weeks=12),
                allocation_interval=timedelta(weeks=1))

request = create_tumult_request(
                session=session,
                query_expr=query_expr,
                budget=RhoZCDPBudget(0.2),
                converter_config=config,
                population_dnf=None,
                population_prob=1.0,
                utility=5)
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Opacus (PyTorch)


In an [Opacus][opacus-url] application, there is a `PrivacyEngine` that encapsulates a standard PyTorch model, optimizer, and data loader. Moreover, the `PrivacyEngine` initializes privacy accounting parameters.


To obtain a Cohere request, we need to define a `ConverterConfig` and subsequently utilize the `create_opacus_request(..)` function, with the necessary inputs.


```Python

# use Opacus to initialize DP model training
model, optimizer, train_loader = PrivacyEngine(accountant="rdp").make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=epochs,
        target_epsilon=3.0,
        target_delta=1e-8,
        max_grad_norm=2.5,
)

# convert to a Cohere request
config = ConverterConfig(
        active_time_window=timedelta(weeks=12), allocation_interval=timedelta(weeks=1))

request = create_opacus_request(
        optimizer=optimizer,
        n_batches_per_epoch=len(train_loader),
        epochs=EPOCHS,
        converter_config=config,
        population_dnf=None,
        population_prob=1.0,
        utility=5)
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## Tests

The `request-adapter` contains a test suite designed to assess the adapter's functionality.

1. Download the adapter test data: [Download (392 MB)](https://drive.google.com/file/d/1qTm48dGh-LAoV31w8RKJm2jcz8rxt6o5/view?usp=sharing)

2. Unarchive the file: `adapter-testdata.zip`

3. Move the result folders to [data](../data/):

    ```sh
    # the directory should look like:

    data/
    ├─ cifar10
    └─ dummy-netflix
    ```

4. Execute the test suite:
    ```sh
    # takes ~37 mins
    poetry run pytest -vvv
    ```



<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->

[python-shield]: https://img.shields.io/badge/python-grey?style=for-the-badge&logo=python
[python-url]: https://www.python.org/

[poetry-shield]: https://img.shields.io/badge/poetry-grey?style=for-the-badge&logo=poetry
[poetry-url]: https://python-poetry.org/


[tumult-shield]: https://img.shields.io/badge/tumult-analytics-grey?style=for-the-badge&logo=apachespark
[tumult-url]: https://www.tmlt.dev/

[opacus-shield]: https://img.shields.io/badge/PyTorch-Opacus-grey?style=for-the-badge&logo=pytorch
[opacus-url]: https://opacus.ai/


[autodp-shield]: https://img.shields.io/badge/autodp-grey?style=for-the-badge&logo=github
[autodp-url]: https://github.com/yuxiangw/autodp
