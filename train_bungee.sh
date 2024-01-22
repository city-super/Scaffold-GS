exp_name='baseline'

voxel_size=0
update_init_factor=128
appearance_dim=0
ratio=1

ulimit -n 4096

./train.sh -d bungeenerf/amsterdam -l ${exp_name} --lod 30 --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} & 
sleep 20s

./train.sh -d bungeenerf/bilbao -l ${exp_name}  --lod 30 --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} &  
sleep 20s

./train.sh -d bungeenerf/hollywood -l ${exp_name} --lod 30 --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} &  
sleep 20s

./train.sh -d bungeenerf/pompidou -l ${exp_name}  --lod 30 --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} &  
sleep 20s

./train.sh -d bungeenerf/quebec -l ${exp_name} --lod 30 --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} &  
sleep 20s

./train.sh -d bungeenerf/rome -l ${exp_name}  --lod 30 --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} &