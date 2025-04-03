from alignment import get_datasets
from trl import SFTTrainer

if __name__ == "__main__":
    # raw_datasets = get_datasets(
    #     {'/local2/wyming/web_sim_traj': 1.0},
    #     splits=['train', 'test'],
    #     columns_to_keep=["messages", "chosen", "rejected", "prompt", "completion", "label"],
    # )
    help(SFTTrainer.__init__)
