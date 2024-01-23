# Hyperparameter Explorer

<a name="readme-top"></a>


<!-- ABOUT THE PROJECT -->
## About The Project

Cohere introduces several hyperparameters that warrant careful consideration in practical
deployments.
Of these, the global target (ϵ, δ) privacy guarantee is certainly the most important hyperparameter.
Yet, the strategies for user rotation and budget unlocking also introduce new hyperparameters, which may seem challenging to configure.
To aid in this process, we provide the hyperparameter explorer, which is a tool for investigating parameter tradeoffs and assessing the suitability of Cohere in different scenarios.



### Built With

* [![Python][python-shield]][python-url]
* [![Poetry][poetry-shield]][poetry-url]
* [![Plotly][plotly-shield]][plotly-url]
* [![Dash][dash-shield]][dash-url]


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

* [Python Poetry](https://python-poetry.org/)

    Install with pipx:
    ```sh
    pipx install poetry
    ```

    or install with the installer script:
    ```sh
    curl -sSL https://install.python-poetry.org | python3 -
    ```

    Ensure that poetry is available on the system's path: `poetry --version`



* [Make](https://www.gnu.org/software/make/)

* Local clone of the repository (with submodules)
    ```sh
    git clone --recurse-submodules git@github.com:pps-lab/cohere.git
    ```


### Installation


1.  Start the local Dash server (from the project root):
    ```sh
    make hyperparameter-explorer
    ```

2. Open the dashboard in the browser at: `http://127.0.0.1:8050/`


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage


At the top of the dashboard, there is a control panel to configure Cohere's hyperparameters:

[![Hyperparameter Controls][hyperparam-controls]](hyperparam-controls)

We visualize the impact of these hyperparameters on three different aspects of the system.


### Active User Distribution

The dashboard visualizes the sample of active users across multiple rounds according to the chosen hyperparameters.
In the provided example, we illustrate the active user sample in four rounds: denoted by the blue dots.
The difference between the active user set of two consecutive rounds is illustrated by green dots, signifying newly added users, and red dots, representing retired users.
The size of the dots is proportional to the fraction of users joining the system at various time intervals.
The x-axis represents the time of users joining the system, whereas the y-axis depicts the round ID in which they participate in the active set.

[![Active User Distribution][hyperparam-user-distribution]](hyperparam-user-distribution)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Budget Unlocking

The dashboard visualizes the budget-unlocking process as described in the paper.
In the example below, we show a scenario with six active groups (K=6), with an equal budget usage of 0.25 per round.
The black area depicts the locked budget, while the white area represents the remaining available budget of the respective group.

[![Budget Unlocking][hyperparam-budget-unlocking]](hyperparam-budget-unlocking)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Mechanism Combinations

Cohere's budget-unlocking strategy ensures that in every round, the available budget consistently falls within a specified range, bounded by minimum and maximum values.
However, due to the non-linear accounting in Renyi Differential Privacy (RDP), it is difficult to understand the possible allocations within this range.

The mechanism combination dashboard facilitates the construction of a composite mechanism by a mixture of multiple fundamental mechanisms.
Subsequently, it displays the frequency with which this particular combination of mechanisms could be allocated per round.


[![Mechanism][hyperparam-mech-combination]](hyperparam-mech-combination)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->

[python-shield]: https://img.shields.io/badge/python-grey?style=for-the-badge&logo=python
[python-url]: https://www.python.org/

[poetry-shield]: https://img.shields.io/badge/poetry-grey?style=for-the-badge&logo=poetry
[poetry-url]: https://python-poetry.org/


[plotly-shield]: https://img.shields.io/badge/plotly-grey?style=for-the-badge&logo=plotly
[plotly-url]: https://plotly.com/


[dash-shield]: https://img.shields.io/badge/dash-grey?style=for-the-badge&logo=plotly
[dash-url]: https://dash.plotly.com/


[hyperparam-controls]: ./../.github/resources/hyperparam-controls.png

[hyperparam-budget-unlocking]: ./../.github/resources/hyperparam-budget-unlocking.png

[hyperparam-user-distribution]: ./../.github/resources/hyperparam-user-distribution.png

[hyperparam-mech-combination]: ./../.github/resources/hyperparam-mech-combination.png
