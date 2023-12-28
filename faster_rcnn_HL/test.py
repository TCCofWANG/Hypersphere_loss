import os
from get_map import test
os.environ["CUDA_VISIBLE_DEVICES"] = '2,3,4,5'
seed = 11
save_dir = '/home/ubuntu/public2/cjp/OD/faster-rcnn-HL/logs'
best_epoch_path = os.path.join(save_dir, "best_epoch_weights_seed%d.pth" % (seed))
test(best_epoch_path, seed)