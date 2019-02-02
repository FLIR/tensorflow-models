#!/bin/bash
#
#   A wrapper for freeze.bash when you only want the latest model frozen
#   see freeze.bash for directory structure
#
chkpts_dir=$1

# Get the latest checkpoint number
latest=$(tail -1 $chkpts_dir/checkpoint )

# rm extra string bits
boiler="all_model_checkpoint_paths: "
boiler=$boiler

latest=${latest#$boiler}
latest=${latest//'"'/''}
latest=${latest#"model.ckpt-"}

echo "Found model checkpoint: $latest"

bash freeze.bash -p $chkpts_dir -c $latest
