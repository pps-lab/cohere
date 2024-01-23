# Reproduce Experiments

<a name="readme-top"></a>


<!-- ABOUT THE PROJECT -->
## About The Project

We provide the necessary commands to reproduce the entire evaluation of the paper.
The evaluation is built using the [DoE-Suite](https://github.com/nicolas-kuechler/doe-suite), making it the most straightforward way to reproduce the results.
However, it is also possible to obtain the individual commands used to invoke the [dp-planner](../dp-planner) and run them manually.

> [!WARNING]
> Executing experiments on your AWS infrastructure involves the creation of EC2 instances, resulting in associated costs.
> It is important to manually check that any created machines are terminated afterward.

### Built With

* [![Poetry][poetry-shield]][poetry-url]
* [![DoE-Suite][doesuite-shield]][doesuite-url]
* [![AWS][aws-shield]][aws-url]
* [![ETHZ Euler][euler-shield]][euler-url]


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

* [Python Poetry](https://python-poetry.org/)
    ```sh
    curl -sSL https://install.python-poetry.org | python3 -
    ```

* [Make](https://www.gnu.org/software/make/)

* Local clone of the repository (with submodules)
    ```sh
    git clone --recurse-submodules git@github.com:pps-lab/cohere.git
    ```

* [TeX Live](https://www.tug.org/texlive/): Only required for reproducing the plots (see e.g., `make plot-all`).
    ```sh
    sudo apt install texlive-full
    ```


### Installation

#### DoE-Suite

1. Setup environment variables for the Doe-Suite:
    ```sh
    # root project directory (expects the doe-suite-config dir in this folder)
    export DOES_PROJECT_DIR=<PATH>

    #  Your unique short name, such as your organization's acronym or your initials.
    export DOES_PROJECT_ID_SUFFIX=<SUFFIX>
    ```

    For AWS EC2:
    ```sh
    export DOES_CLOUD=aws

    # name of ssh key used for setting up access to aws machines (name of key not path)
    export DOES_SSH_KEY_NAME=<YOUR-PRIVATE-SSH-KEY-FOR-AWS>
    ```

    For ETHZ Euler (Slurm-based Scientific Compute Cluster):
    ```sh
    export DOES_CLOUD=euler

    # Replace <YOUR-NETHZ> with your NETHZ username
    export DOES_EULER_USER=<YOUR-NETHZ>
    ```


2. Set up SSH Config and for AWS setup AWS CLI.
Currently, the `doe-suite` is configured to use the AWS region `eu-central-1`.
For more details refer to the [doe-suite documentation](https://nicolas-kuechler.github.io/doe-suite/installation.html#base-installation).


[!Tip]
To debug problems it can be helpful to comment out the line `stdout_callback = community.general.selective` in [doe-suite/ansible.cfg](../doe-suite/ansible.cfg).


#### Paper Results and Workload Data

1. Download the raw results of the Cohere evaluation: [Download (3.5 GB)](https://drive.google.com/uc?export=download&id=1Bl2JE3KUV5cBpsoMdxV1LgVl1e0EkIFw)


2. Unarchive the file: `sp24-results-reduced.zip`


3. Move the result folders to [doe-suite-results](../doe-suite-results/):

    ```sh
    # the directory should look like:

    doe-suite-results/
    ├─ sp_sub_1690826136
    ├─ sp_unlock_1690743687
    └─ sp_workloads_1690805680
    ```

4. The `sp24-results-reduced.zip` includes also an archive labeled `applications.zip`, containing the original workloads. Alternatively, these workloads can be newly sampled with the [workload-simulator](../workload-simulator/).


5. The [doe-suite-config/roles/data-setup](roles/data-setup) Ansible role is responsible for making the `applications.zip` accessible within the remote experiment environment.
The role supports two options to obtain the `applications.zip`:
    1. **Download from your AWS S3**:

        - Ensure that the AWS CLI is installed. (The data-setup role uses the AWS CLI to create a temporary download link.)

        - Create an S3 bucket named `privacy-management` within your AWS S3 account (in `eu-central-1` region).

        - Upload the `applications.zip` file into this bucket.

        - Ensure that the `download_data_from_aws` variable in [doe-suite-config/group_vars/all/main.yml](group_vars/all/main.yml) is set to `True` to enable the download from AWS.

    2. **Manual Placement** (only for Euler):

        - Copy the `applications.zip` file to the designated `{{ data_dir }}` folder as specified in [doe-suite-config/group_vars/all/main.yml](group_vars/all/main.yml).

        - Set the `download_data_from_aws` variable in [doe-suite-config/group_vars/all/main.yml](group_vars/all/main.yml) is set to `False` to use the already available `applications.zip` rather than trying to download it from S3.




#### Gurobi License

The [dp-planner](../dp-planner) relies on [Gurobi](https://www.gurobi.com/) for solving resource allocation problems, which requires a Gurobi license.
Free academic licenses can be obtained from the following link: https://www.gurobi.com/downloads/end-user-license-agreement-academic/

The acquired license keys still need to be activated on the designated machine.
To facilitate this process, we provide the Ansible role [doe-suite-config/roles/setup-gurobi](roles/setup-gurobi), which installs a valid Gurobi license on the remote experiment environment.
While ETH Euler already possesses a valid license, specific actions are required for AWS instances due to the necessity of individual licenses per EC2 instance ([see additional info](https://support.gurobi.com/hc/en-us/articles/360031842772-How-do-I-configure-an-AWS-EC2-instance-so-the-license-file-remains-valid-)).


For AWS EC2:
1. Set the desired license keys in [doe-suite-config/group_vars/all/gurobi_license.yml](group_vars/all/gurobi_license.md):

    ```yaml
    gurobi_license_grbgetkey:
    - <license_key_1>
    - <license_key_2>
    ```

    Note, that we provide a [gurobi_license.md](group_vars/all/gurobi_license.md) template file in the same folder which should then be filled in and renamed to `gurobi_license.yml`.

2. To activate an academic license, the license request must originate from within an academic domain list.
To facilitate this, the `doe-suite` needs to establish a temporary VPN connection for license acquisition.
We use the [OpenConnect VPN client](https://www.infradead.org/openconnect/) to establish a connection to the ETH Zurich VPN. For other domains, you may need to adjust the responsible task in  [doe-suite-config/roles/setup-gurobi/tasks/aws.yml](roles/setup-gurobi/tasks/aws.yml):

    ```yaml
    - name: Connect to VPN (if this fails, check credentials)
        expect:
        command: "sudo openconnect sslvpn.ethz.ch -s '/usr/local/bin/vpn-slice gurobi.com apps.gurobi.com portal.gurobi.com'"
        responses:
            GROUP:
            - "student-net"
            Username:
            - "{{ vpn_username }}"
            Password:
            - "{{ vpn_password }}"
        async: 600
        poll: 0
        no_log: true
        when: lic_1.stat.islnk is not defined
    ```

    This task references `{{ vpn_username }}` and `{{ vpn_password }}`, both of which are part of the template [doe-suite-config/group_vars/all/gurobi_license.yml](group_vars/all/gurobi_license.md):

    ```yaml
    vpn_username: <vpn_username>

    vpn_password: <vpn_password>
    ```




<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Experiments


After the installation, the original experiment outcomes referenced in the paper will be accessible at [doe-suite-results](../doe-suite-results).
Any additional experiments conducted will also be stored in the same location.


For simplifying the reproduction of results, we provide a custom [Makefile](../Makefile) to simplify the interface.
All further commands available within the `doe-suite` can be accessed via the [Makefile](../doe-suite/Makefile) located in [doe-suite](../doe-suite/).
For further details, please refer to the [doe-suite documentation](https://nicolas-kuechler.github.io/doe-suite/).


We can reconstruct all evaluation figures in the paper with:
```
make plot-all
```

### Benefits of Subsampling

The suite design [doe-suite-config/designs/sp_sub.yml](designs/sp_sub.yml) defines the experiment.


To obtain all individual commands, use:
```
make cmd-subsampling
```


For rerunning all the experiments, execute:
```
make run-subsampling
```


The generation of the subsampling plot (Figure 5) relies on the super ETL config [doe-suite-config/super_etl/sp_sub.yml](super_etl/sp_sub.yml).

To regenerate the plot, use:
```
make plot-subsampling
```

By default, the figure is derived from the experiment results shown in the paper.
However, it is possible to provide a custom results directory obtained from `run-subsampling`.

<div align="center">
    <img src=./../.github/resources/subsampling-fig5.png width="600">
</div>


<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Benefits of our Budget Unlocking

The suite designs [doe-suite-config/designs/sp_unlock.yml](designs/sp_unlock.yml) and [doe-suite-config/designs/sp_workloads.yml](designs/sp_workloads.yml) define the experiments.


To obtain all individual commands, use:
```
make cmd-unlocking
```


For rerunning all the experiments, execute:
```
make run-unlocking
```


The generation of the unlocking plot (Figure 6) relies on the super ETL config [doe-suite-config/super_etl/sp_workloads.yml](super_etl/sp_workloads.yml).

To regenerate the plot, use:
```
make plot-unlocking
```

By default, the figure is derived from the experiment results shown in the paper.
However, it is possible to provide a custom results directory obtained from `run-unlocking`.


<div align="center">
    <img src=./../.github/resources/unlocking-fig6.png width="600">
</div>


<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Comparison to PrivateKube

The suite design [doe-suite-config/designs/sp_workloads.yml](designs/sp_workloads.yml) defines the experiment.


To obtain all individual commands, use:
```
make cmd-comparison
```


For rerunning all the experiments, execute:
```
make run-comparison
```

The generation of the comparison plot (Figure 7) relies on the super ETL config [doe-suite-config/super_etl/sp_workloads.yml](super_etl/sp_workloads.yml).

To regenerate the plot, use:
```
make plot-comparison
```

By default, the figure is derived from the experiment results shown in the paper.
However, it is possible to provide a custom results directory obtained from `run-comparison`.



<div align="center">
    <img src=./../.github/resources/comparison-fig7.png width="1200">
</div>


<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->

[poetry-shield]: https://img.shields.io/badge/poetry-grey?style=for-the-badge&logo=poetry
[poetry-url]: https://python-poetry.org/


[doesuite-shield]: https://img.shields.io/badge/doe--suite-grey?style=for-the-badge&logo=github
[doesuite-url]: https://github.com/nicolas-kuechler/doe-suite


[aws-shield]: https://img.shields.io/badge/aws-ec2-grey?style=for-the-badge&logo=amazonaws
[aws-url]: https://aws.amazon.com/


[euler-shield]: https://img.shields.io/badge/ethz-euler-grey?style=for-the-badge
[euler-url]: https://scicomp.ethz.ch/wiki/Euler
