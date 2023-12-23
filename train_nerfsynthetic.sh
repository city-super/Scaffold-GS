exp_name='baseline_v0.001_init4'

voxel_size=0.001
update_init_factor=4


ulimit -n 4096

./train.sh -d nerfsynthetic/chair -l ${exp_name}  --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} & 
sleep 20s

./train.sh -d nerfsynthetic/drums -l ${exp_name}  --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} & 
sleep 20s

./train.sh -d nerfsynthetic/ficus -l ${exp_name}  --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} & 
sleep 20s

./train.sh -d nerfsynthetic/hotdog -l ${exp_name}  --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} & 
sleep 20s

./train.sh -d nerfsynthetic/lego -l ${exp_name}  --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} & 
sleep 20s

./train.sh -d nerfsynthetic/materials -l ${exp_name}  --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} & 
sleep 20s

./train.sh -d nerfsynthetic/mic -l ${exp_name}  --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} & 
sleep 20s

./train.sh -d nerfsynthetic/ship -l ${exp_name}  --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} & 
# sleep 20s