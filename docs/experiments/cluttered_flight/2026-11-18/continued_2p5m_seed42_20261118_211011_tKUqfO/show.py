from continue_ppo import CHECKPOINTS, RUN

import evaluate as baseline_eval
from view import view

baseline_eval.CHECKPOINTS.update(CHECKPOINTS)
baseline_eval.RUN = RUN

if __name__ == "__main__":
    view()
