exp_name='baseline'

voxel_size=0
update_init_factor=16
appearance_dim=32
ratio=1

ulimit -n 4096

./train.sh -d imw2020/brandenburg_gate -l ${exp_name} --lod 30 --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} & 
sleep 20s

./train.sh -d imw2020/buckingham_palace -l ${exp_name}  --lod 30 --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} &  
sleep 20s

./train.sh -d imw2020/colosseum_exterior -l ${exp_name} --lod 30 --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} &  
sleep 20s

./train.sh -d imw2020/florence_cathedral_side -l ${exp_name}  --lod 30 --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} &  
