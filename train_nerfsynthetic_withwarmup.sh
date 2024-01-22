exp_name='baseline_warmup'

voxel_size=0.0
update_init_factor=4
appearance_dim=0
ratio=1
warmup='True'

ulimit -n 4096

./train.sh -d nerfsynthetic/chair -l ${exp_name}  --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --warmup ${warmup} --appearance_dim ${appearance_dim} --ratio ${ratio} & 
sleep 20s

./train.sh -d nerfsynthetic/drums -l ${exp_name}  --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --warmup ${warmup} --appearance_dim ${appearance_dim} --ratio ${ratio} & 
sleep 20s

./train.sh -d nerfsynthetic/ficus -l ${exp_name}  --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --warmup ${warmup} --appearance_dim ${appearance_dim} --ratio ${ratio} & 
sleep 20s

./train.sh -d nerfsynthetic/hotdog -l ${exp_name}  --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --warmup ${warmup} --appearance_dim ${appearance_dim} --ratio ${ratio} & 
sleep 20s

./train.sh -d nerfsynthetic/lego -l ${exp_name}  --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --warmup ${warmup} --appearance_dim ${appearance_dim} --ratio ${ratio} & 
sleep 20s

./train.sh -d nerfsynthetic/materials -l ${exp_name}  --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --warmup ${warmup} --appearance_dim ${appearance_dim} --ratio ${ratio} & 
sleep 20s

./train.sh -d nerfsynthetic/mic -l ${exp_name}  --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --warmup ${warmup}  --appearance_dim ${appearance_dim} --ratio ${ratio} & 
sleep 20s

./train.sh -d nerfsynthetic/ship -l ${exp_name}  --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --warmup ${warmup} --appearance_dim ${appearance_dim} --ratio ${ratio} & 
# sleep 20s