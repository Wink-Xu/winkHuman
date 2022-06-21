## run
-- image
python pipeline.py --config config/infer_cfg.yml --source=test_image.jpg --device=1 


-- video
python pipeline.py --config config/infer_cfg.yml --source=test_video.mp4 --device=1 
## unit_test  detector
python detector/detect.py --image_file=test_image.jpg --device=0 --model_dir=detector/model_dir/crowdhuman_yolov5m.pt




# todo list
1. yolov5训自己的数据集。 和张帆对比。（yolo系列（包括yolox）要学会了，到可以被提问的程度。也看下centernet）
2. pipleline 跑视频 完成跟踪。（deepsort） 代码整理。加注释。
3. 行为识别ST-GCN，成功跑demo，inference。
4. 集成attribute
5. 集成ReID
6. 集成HRNet
7. 集成ST-GCN
8. tensort部署
9. 服务化