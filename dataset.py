import torch
from saliency_db import saliency_db
from transforms import TemporalRandomCrop, SpatialTransform, SpatialTransform_norm

def get_dataloader(root:str, mode:str, task:str,
				   datasetName_list:list=None, 
				   batch_size:int=8, num_workers:int=4,
				   sample_size:int=112,			# 输入尺寸
				   sample_duration:int=16,		# 最终采样长度
				   step_duration:int=90,		# 空余采样长度
				   spatial_transform=None,
				   spatial_transform_norm=None,
				   temporal_transform=None,
				   pin_memory:bool=True,
				   infoBatch_epoch:int=-1,		# is use infoBatch
				   ):
	
	assert mode in ['train', 'val', 'test'], f"mode should be 'train', 'val' or 'test', but got {mode}"
	assert task in ['class_increase', 'domain_increase'], f"task should be 'class_increase' or 'domain_increase', but got {task}"
	
	if task == 'class_increase':
		if datasetName_list == None:
			datasetName_list = ['DIEM', 'Coutrot_db1', 'Coutrot_db2', 'SumMe', 'ETMD_av', 'AVAD']
		else:
			for i in datasetName_list:
				assert i in ['DIEM', 'Coutrot_db1', 'Coutrot_db2', 'SumMe', 'ETMD_av', 'AVAD'], f"datasetName_list should be ['DIEM', 'Coutrot_db1', 'Coutrot_db2', 'SumMe', 'ETMD_av', 'AVAD'], but got {datasetName_list}"
	elif task == 'domain_increase':
		if datasetName_list == None:
			datasetName_list = ['doing', 'nature', 'society', 'something', 'talk']
		else:
			for i in datasetName_list:
				assert i in ['doing', 'nature', 'society', 'something', 'talk'], f"datasetName_list should be ['doing', 'nature', 'society', 'something', 'talk'], but got {datasetName_list}"

	if spatial_transform == None:
		spatial_transform = SpatialTransform(mode, sample_size)
	if spatial_transform_norm == None:
		spatial_transform_norm = SpatialTransform_norm((0.3818, 0.3678, 0.3220), (0.2727, 0.2602, 0.2568))
	if temporal_transform == None:
		temporal_transform = TemporalRandomCrop(sample_duration)


	txt_mode = 'train' if mode == 'train' else 'test' 
	all_dataset = []

	for dataset_name in datasetName_list:
		print(f"load dataset: {dataset_name}({mode})...")

		if task == 'class_increase':
			training_data = saliency_db(
				f'{root}/video_frames/{dataset_name}/',
				f'{root}/fold_lists/{dataset_name}_list_{txt_mode}_1_fps.txt',
				f'{root}/annotations/{dataset_name}/',
				f'{root}/video_audio/{dataset_name}/',
				spatial_transform=spatial_transform,
				spatial_transform_norm=spatial_transform_norm,
				temporal_transform= temporal_transform,
				exhaustive_sampling=(mode == 'test'),
				sample_duration=sample_duration,	# 采样长度
				step_duration=step_duration,		# 窗口长度
				)

		elif task == 'domain_increase':
			training_data = saliency_db(
				f'{root}/video_frames/',
				f'{root}/class_category/{dataset_name}_{txt_mode}.txt',
				f'{root}/annotations/',
				f'{root}/video_audio/',
				spatial_transform=spatial_transform,
				spatial_transform_norm=spatial_transform_norm,
				temporal_transform= temporal_transform,
				exhaustive_sampling=(mode == 'test'),
				sample_duration=sample_duration,	# 采样长度
				step_duration=step_duration,		# 窗口长度
				)
		
		all_dataset.append(training_data)
	
	data_set = torch.utils.data.ConcatDataset(all_dataset)

	if infoBatch_epoch>0:		# TODO: need to test
		from .utils.InfoBatch import InfoBatch
		data_set = InfoBatch(data_set, num_epochs=infoBatch_epoch)
		print("*** use infoBatch! Remember use \"loss_fun = nn.CrossEntropyLoss(reduction='none')\" and \"loss = dataset.update(loss)\" in training loop")


	data_loader = torch.utils.data.DataLoader(
		data_set,
		batch_size=batch_size,
		num_workers=num_workers,
		shuffle=(mode == 'train' and infoBatch_epoch<=0),
		pin_memory=pin_memory,
		drop_last=(mode == 'train'),
		sampler = data_set.sampler if infoBatch_epoch>0 else None)

	if infoBatch_epoch>0:
		return data_loader, data_set
	else:
		return data_loader
