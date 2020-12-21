# python train.py --img 416 --batch 32 --epochs 250 --data data/coco_fangweisui.yaml --weights weights/yolov5s.pt --cfg models/yolov5s_fangweisui.yaml
python train.py --img 640 --batch 8 --epochs 100 --data ./data/hand.yaml --cfg ./models/yolov5s_hand.yaml --weights weights/yolov5s.pt --name s_hand100
