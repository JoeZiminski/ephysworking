from dammy_prepro_script import preprocess_single_subject

# Basic script to run `dammy_prepro_script.py`.

sub_name = "DO81"  # "DO81"

session_paths = [
 #   "/ceph/neuroinformatics/neuroinformatics/scratch/jziminski/ephys/dammy/DO79_2024-02-16_16-47-57_001/Record Node 101/experiment1/recording1",
  #  "/ceph/neuroinformatics/neuroinformatics/scratch/jziminski/ephys/dammy/DO79_2024-02-16_18-55-52_002/Record Node 101/experiment1/recording1"
    "/ceph/neuroinformatics/neuroinformatics/scratch/jziminski/ephys/dammy/DO81/DO81_2024-04-25_001/Record Node 101/experiment1/recording1",
    "/ceph/neuroinformatics/neuroinformatics/scratch/jziminski/ephys/dammy/DO81/DO81_2024-04-25_002/Record Node 101/experiment1/recording1",
    "/ceph/neuroinformatics/neuroinformatics/scratch/jziminski/ephys/dammy/DO81/DO81_2024-04-25_003/Record Node 101/experiment1/recording1",
]

output_path = "/ceph/neuroinformatics/neuroinformatics/scratch/jziminski/ephys/dammy/motion_compare"

motion_method = "nonrigid_fast_and_accurate" # "kilosort_like"  # nonrigid_accurate  "nonrigid_fast_and_accurate"

preprocess_single_subject(sub_name, session_paths, output_path, motion_method, n_jobs=50)