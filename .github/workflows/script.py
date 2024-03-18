import sys

sys.path.append(".")

import covalent_cloud as cc
from finrl_workflow import workflow


if __name__ == "__main__":
    RUNID_FILE = "runid_status.csv"
    RESULTS_FILE = "results.csv"
    VOLUME_NAME = "/finrl"

    TRAIN_START_DATE = '2009-01-01'
    TRAIN_END_DATE = '2020-07-01'
    TRADE_START_DATE = '2020-07-01'
    TRADE_END_DATE = '2021-10-29'

    # Dispatch the workflow to the Covalent Cloud
    runid = cc.dispatch(workflow, volume=VOLUME_NAME)(
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
    print(f"Dispatched runid: {runid} Submitted")

    # # Update the runid status to PENDING
    # update_runid(runid, Status.PENDING, runid_file=RUNID_FILE)
    # print(f"Runid: {runid} Status: {Status.PENDING.value} Updated")

    # # Check and update the runid status
    # check_and_update_status(
    #     runid_file=RUNID_FILE,
    #     results_file=RESULTS_FILE,
    # )
    # print(
    #     f"Old runid status updated in {RUNID_FILE} and results written to {RESULTS_FILE}"
    # )
