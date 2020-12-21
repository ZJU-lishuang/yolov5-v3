# python train_sparsity.py --img 416 --batch 16 --epochs 5 --data data/coco_fangweisui.yaml --cfg models/yolov5s_fangweisui.yaml --weights weights/last.pt --name s -sr --s 0.001 --prune 1
python train_sparsity.py --img 640 --batch 16 --epochs 150 --data data/hand.yaml --cfg models/yolov5s_hand.yaml --weights weights/last_s_hand.pt --name s_to_prune -sr --s 0.001 --prune 1
