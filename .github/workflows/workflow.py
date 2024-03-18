import os
import itertools
import pandas as pd

import covalent as ct
import covalent_cloud as cc
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.config import INDICATORS
from finrl import config_tickers
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3 import DDPG, PPO


cc.save_api_key(os.environ["CC_API_KEY"])
cc.create_env(
    name="fin-rl",
    conda={
        "channels": ["conda-forge", "defaults"],
        "dependencies": [
            "python=3.10", "gcc", "gxx", "cxx-compiler", "swig"
        ]
    },
    pip=[
        "numpy==1.23.5",
        "git+https://github.com/AI4Finance-Foundation/FinRL.git",
        "stable-baselines3==2.2.1"
    ],
    wait=True
)
cpu_executor = cc.CloudExecutor(
    env="fin-rl",
    num_cpus=4,
    memory=16384,
    time_limit=60*60
)
gpu_executor = cc.CloudExecutor(
    # num_gpus=1,
    env="fin-rl",
    # gpu_type="v100",
    memory=16384,
    num_cpus=4,
    time_limit=60*60
)
# large_cpu_exec = gpu_exec
# default_exec = 'local'
VOLUME_NAME = "/finrl"


@ct.electron(executor=cpu_executor)
def download_stock_data(start_date, end_date):
    df_raw = YahooDownloader(
        start_date=start_date,
        end_date=end_date,
        ticker_list=config_tickers.DOW_30_TICKER
    ).fetch_data()

    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=True,
        use_turbulence=True,
        user_defined_feature=False
    )
    processed = fe.preprocess_data(df_raw)

    list_ticker = processed["tic"].unique().tolist()
    list_date = list(
        pd.date_range(
            processed['date'].min(),
            processed['date'].max()
        ).astype(str)
    )
    combination = list(
        itertools.product(list_date, list_ticker)
    )
    processed_full = pd.DataFrame(
        combination, columns=["date", "tic"]
    ).merge(processed, on=["date", "tic"], how="left")

    processed_full = processed_full[
        processed_full['date'].isin(processed['date'])
    ]
    processed_full = processed_full.sort_values(['date', 'tic'])
    processed_full = processed_full.fillna(0)
    # save to volume
    save_path = Path(VOLUME_NAME) / "stock_data.csv"
    processed_full.to_csv(save_path, index=False)
    return save_path


@ct.electron(executor=gpu_executor)
def train_rl_model(train_data_path, rl_algorithm='ddpg', alg_params={}):
    train = pd.read_csv(train_data_path)
    train = train.set_index(train.columns[0])
    train.index.names = ['']
    stock_dimension = len(train.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }
    e_train_gym = StockTradingEnv(df=train, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()

    agent = DRLAgent(env=env_train)
    model_rl = agent.get_model(
        rl_algorithm, model_kwargs=alg_params.get('model_kwargs', {})
    )

    timesteps = 5 if 'timesteps' not in alg_params else alg_params['timesteps']
    trained_agent = agent.train_model(
        model=model_rl, tb_log_name=rl_algorithm,
        total_timesteps=timesteps
    )
    # save model
    TRAINED_MODEL_DIR = Path(VOLUME_NAME) / "models"
    TRAINED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_folder = f"agent_{rl_algorithm}"
    model_save_path = TRAINED_MODEL_DIR / model_folder
    trained_agent.save(model_save_path)
    return model_save_path


@ct.electron(executor=gpu_executor)
def backtest(trained_model_path, test_data_path):
    test_data = pd.read_csv(test_data_path)
    test_data = test_data.set_index(test_data.columns[0])
    test_data.index.names = ['']
    stock_dimension = len(test_data.tic.unique())

    state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }

    e_trade_gym = StockTradingEnv(
        df=test_data, turbulence_threshold=70,
        risk_indicator_col='vix', **env_kwargs
    )
    # posix path to string
    if "ppo" in str(trained_model_path):
        trained_model = PPO.load(trained_model_path)
    elif "ddpg" in str(trained_model_path):
        trained_model = DDPG.load(trained_model_path)

    df_account_value, df_actions = DRLAgent.DRL_prediction(
        model=trained_model,
        environment=e_trade_gym
    )
    df_result = (
        df_account_value.set_index(df_account_value.columns[0])
    )
    return df_result


@ct.electron(executor=cpu_executor)
def split_data_by_date(
    data_path, start_date, end_date, output_file_name="train_data.csv"
):
    data_full = pd.read_csv(data_path)
    data_part = data_split(
        data_full, start_date, end_date
    )
    save_path = Path(VOLUME_NAME) / output_file_name
    data_part.to_csv(save_path)
    return save_path


@ct.electron(executor=cpu_executor)
def get_dji_data(start_date, end_date):
    df_dji = YahooDownloader(
        start_date=start_date,
        end_date=end_date, ticker_list=["dji"]
    ).fetch_data()
    df_dji = df_dji[["date", "close"]]
    fst_day = df_dji["close"][0]
    dji = pd.merge(
        df_dji["date"], df_dji["close"].div(fst_day).mul(1000000),
        how="outer", left_index=True, right_index=True,
    ).set_index("date")
    dji.columns = ["dji"]
    return dji


@ct.electron(executor=cpu_executor)
def plot_results(performance_data):
    save_folder = Path(VOLUME_NAME) / "graphs"
    save_folder.mkdir(parents=True, exist_ok=True)

    df = pd.concat(performance_data, axis=1)
    # start new plot
    plt.clf()
    # increase figure size
    plt.figure(figsize=(15, 5))
    plt.title('Portfolio Value over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (USD)')
    # add grid
    plt.grid()

    plt.plot(
        df.index, df['dji'],
        label="Dow Jones Industrial Average", color='black', linestyle='dashed'
    )
    plt.plot(df.index, df['ppo'], label="PPO", color='red')
    plt.plot(df.index, df['ddpg'], label="DDPG", color='green')

    desired_ticks = df.index[::int(len(df.index) / 5)]
    plt.xticks(desired_ticks, labels=desired_ticks)
    legend_labels = ["DDPG", "PPO", "Dow Jones Industrial Average"]
    plt.legend()

    # save plot
    plt.savefig(save_folder / "portfolio_value.png")
    # open and return image
    return Image.open(save_folder / "portfolio_value.png")


@ct.lattice(workflow_executor=gpu_executor, executor=gpu_executor)
def workflow(
    train_start_date, train_end_date,
    test_start_date, test_end_date,
    algorithm_params={}
):
    data_path = download_stock_data(
        train_start_date, test_end_date
    )
    train_data_path = split_data_by_date(
        data_path, train_start_date, train_end_date, "train_data.csv"
    )
    test_data_path = split_data_by_date(
        data_path, test_start_date, test_end_date, "test_data.csv"
    )
    results = []
    for algorithm in algorithm_params.keys():
        trained_model_path = train_rl_model(
            train_data_path, algorithm, algorithm_params.get(algorithm, {})
        )
        result = backtest(trained_model_path, test_data_path)
        result.columns = [algorithm]
        results.append(result)

    results.append(get_dji_data(test_start_date, test_end_date))

    image_plot = plot_results(results)
    ct.wait(image_plot, results)
    return image_plot


if __name__ == '__main__':
    # Define the start and end dates for the data
    TRAIN_START_DATE = '2009-01-01'
    TRAIN_END_DATE = '2020-07-01'
    TRADE_START_DATE = '2020-07-01'
    TRADE_END_DATE = '2021-10-29'
    # plot_figure = workflow(
    #     TRAIN_START_DATE, TRAIN_END_DATE,
    #     TRADE_START_DATE, TRADE_END_DATE,
    #     algorithm_params={
    #         "ddpg": {"timesteps": 5, "model_kwargs": {
    #             "device": "cuda:0", "batch_size": 2048
    #         }},
    #         "ppo": {"timesteps": 5, "model_kwargs": {"device": "cuda:0"}}
    #     }
    # )
    # plot_figure.save('portfolio_value.png')

    # plt.savefig(save_folder / "portfolio_value.png")
    volume = cc.volume(VOLUME_NAME)
    dispatch_id = cc.dispatch(workflow, volume=volume)(
        TRAIN_START_DATE, TRAIN_END_DATE,
        TRADE_START_DATE, TRADE_END_DATE,
        algorithm_params={
            "ddpg": {
                "timesteps": 5, "model_kwargs": {
                    "batch_size": 2048
                }
            },
            "ppo": {
                "timesteps": 5, "model_kwargs": {}
            }
        }
    )
    # results = cc.get_result(dispatch_id)
