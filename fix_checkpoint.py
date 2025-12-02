import torch
checkpoint = torch.load('checkpoints/epoch=0-step=4095.ckpt')

print(checkpoint['dataset_state_dict'])
checkpoint['dataset_state_dict']['pq_idx'] = torch.tensor([[4], [4], [4], [4]])
checkpoint['dataset_state_dict']['rg_idx'] = torch.tensor([[43], [43], [43], [43]])
print(checkpoint['dataset_state_dict'])

torch.save(checkpoint, 'checkpoints/epoch=0-step=4095-mod.ckpt')
