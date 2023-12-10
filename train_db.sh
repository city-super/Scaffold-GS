exp_name='baseline'

voxel_size=0.005
update_init_factor=16

ulimit -n 4096

./train.sh -d blending/playroom -l ${exp_name} --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} & 
sleep 20s

./train.sh -d blending/drjohnson -l ${exp_name} --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} & 