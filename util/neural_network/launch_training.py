
import os
import sys

# Allow running as a script: python util/neural_network/launch_training.py ...
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from util.neural_network.nn_model import NeuralNet
from util.neural_network.train_nn import train_nn
from util.tools.basic import load_dict


def launch_training(network_params, optimizer_params, training_params):
    network_params["output_type"] = "rri"
    
    print("Training RRI model...")

    
    print(f"training model 1:  Linear Resistor")
    network_params["target_coef_ind"] = 0
    network_params["layer_width"] = 40
    network_params["num_layers"] = 1
    training_params["num_epochs"] = 500
    optimizer_params["decay_rate"] = 0.5
    optimizer_params["init"] = 0.01
    model = NeuralNet(network_params, optimizer_params)
    train_nn(model, training_params)

    print(f"training model 2:  Stenosis Resistor")
    network_params["target_coef_ind"] = 1
    model = NeuralNet(network_params, optimizer_params)
    train_nn(model, training_params)

    print(f"training model 3:  Inductor")
    network_params["target_coef_ind"] = 2
    model = NeuralNet(network_params, optimizer_params)
    train_nn(model, training_params)
    return

if __name__ == "__main__":
    set_name = sys.argv[1]
    num_geos = int(sys.argv[2])
    output_type = "rri"#, "ri", or "rr"
    set_type = "test"
    #pdb.set_trace()

    split_ind_dict = load_dict(
        f"data/split_indices/{set_name}/{set_type}/train_val_ind_{set_name}_num_geos_{num_geos}"
    )

    network_params = {"num_input_features": 22,
                      "num_layers": 1,
                      "layer_width":20,
                      "output_type": output_type,
                      "set_name": set_name,
                      "set_type": set_type,
                      "num_geos": num_geos,
                      "data_root": "data",
                      "pred_mode": "m1"}
    
    training_params = {"num_epochs": 1000, 
                       "batch_size": 1,
                       "train_inds": split_ind_dict["train_ind"],
                       "val_inds": split_ind_dict["val_ind"],
                       "num_offsets": split_ind_dict["num_offsets"],}
    
    optimizer_params = {#"step_size": 0.0002,
                        "init" : 0.02,
                        "transition_steps": 1000,
                        "decay_rate" : 0.95}
    
    launch_training(network_params, optimizer_params, training_params)