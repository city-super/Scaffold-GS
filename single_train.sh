scene='tandt/truck'
exp_name='baseline'
voxel_size=0.01
update_init_factor=16
gpu=-1

# example:
./train.sh -d ${scene} -l ${exp_name} --gpu ${gpu} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor}