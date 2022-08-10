# De-anonymizing Tornado Cash

## Headers

### tornado_deposit.csv

|    ts      |   txhash   | address | tornado_cash_address | value | gas_price
|  ----      | ----      | -----------| ------------------- | ---| --- |

### tornado_withdraw.csv

|    ts      |   txhash   | address | tornado_cash_address | value | gas_price
|  ----    | ----      | -----------| ------------------- | ---| --- |

### tornado_related_txs/depositor(withdrawer)/*.csv

|    txhash   |   from_address   | to_address | value
|  ----    | ----      | -----------| ------------------- |

## Environment Setup

1. Prerequisite installation
    - [pyenv](https://github.com/pyenv/pyenv)
    - [pipenv](https://github.com/pypa/pipenv)
    - [docker compose](https://docs.docker.com/compose/install/)
    - tornado_deposit.csv
    - tornado_withdraw.csv
    - tornado_related_txs/
        - depositor
        - withdrawer
2. install python

    ```bash
    pyenv install --python 3.10.4
    ```

3. switch to virtual environment

    ```bash
    pipenv install
    ```

4. open vitual environment shell

    ```bash
    pipenv shell
    ```

5. pull `Nebula Graph` image

    ```bash
    git clone -b release-3.2 https://github.com/vesoft-inc/nebula-docker-compose.git

    ```

6. start `Nebula Graph` service

    ```bash
    cd nebula-docker-compose/ && docker compose up -d
    ```

7. access `nebula console`

    ```bash
    docker exec -it nebula-docker-compose-console-1 nebula-console -u root -p nebula --address=graphd --port=9669
    ```

8. [optional] download `Nebula Studio (UI)`

    - 8.1

        ```bash
        mkdir nebula-studio && cd nebula-studio
        ```

    - 8.2

        ```bash
        wget -c https://oss-cdn.nebula-graph.com.cn/nebula-graph-studio/3.3.2/nebula-graph-studio-3.3.2.tar.gz
        ```

    - 8.3

        ```bash
        tar -zxvf nebula-graph-studio-3.3.2.tar.gz -C .
        ```

    - 8.4

        ```bash
        docker compose up -d
        ```

    - 8.5
        Access [Nebula Studio](http://localhost:7001)

    - 8.6
        |          |   value   |
        |  ----    | ----      |
        | host     | echo $(hostname -I \| cut -d ' ' -f1):9669 |
        | username | root      |
        | password | nebula    |
9. [do it once] init `Nebula Graph` via console or UI

    execute content in `script/insert_txs_nebula.ngql`

10. create `.env` and fill in

    ```.env
    w3_provider=<your web3 provider>
    address=<your ip>
    port="9669" or <your port>
    username="root" or <your username>
    password="nebula" or <your password>
    ```

11. put your csv into `data` folder

    ```tree
    📦data
    ┣ 📂tornado_related_txs
    ┃ ┣ 📂depositor
    ┃ ┃ ┣ 📜depositor_2020-06-17.csv
    ┃ ┃ ┣ 📜depositor_2021-01-01.csv
    ┃ ┃ ┣ 📜depositor_2022-01-01.csv
    ┃ ┃ ┗ 📜depositor_2022-06-01.csv
    ┃ ┗ 📂withdrawer
    ┃ ┃ ┣ 📜withdrawer_2020-06-16.csv
    ┃ ┃ ┣ 📜withdrawer_2020-12-31.csv
    ┃ ┃ ┣ 📜withdrawer_2021-05-31.csv
    ┃ ┃ ┗ 📜withdrawer_2021-12-31.csv
    ┣ 📜tornado_deposit.csv
    ┣ 📜tornado_withdraw.csv
    ```

12. execute main

    ```bash
    python main.py
    ```
