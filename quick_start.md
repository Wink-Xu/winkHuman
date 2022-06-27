## run
-- image
python pipeline.py --config config/infer_cfg.yml --source=test_image.jpg --device=1 

-- video
python pipeline.py --config config/infer_cfg.yml --source=test_video.mp4 --device=1 

-- video
python pipeline.py --config config/infer_cfg.yml --source=test_video.mp4 --device=1 --enable_action True
## unit_test  detector
python detector/detect.py --image_file=test_image.jpg --device=0 --model_dir=detector/model_dir/crowdhuman_yolov5m.pt

## unit_test  keypoint
python keypoint/keypoint.py --image_file=body_image.jpg --device=0 --model_dir=keypoint/model_dir

