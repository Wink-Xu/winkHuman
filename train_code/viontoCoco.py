import os
import shutil

def xyxy2xywh(anchor):
    cx = anchor[0] + (anchor[2] - anchor[0])//2
    cy = anchor[1] + (anchor[3] - anchor[1])//2
    w  = anchor[2] - anchor[0]
    h = anchor[3] - anchor[1]
    return cx,cy,h,w 

main_txt = '/home2/xuzihao/zzs/overlook_44w/label.txt'
main_dir = '/home2/xuzihao/zzs/overlook_44w'
output_dir_train = '/home2/xuzihao/zzs/overlook_44w_likeCoco/train'
output_dir_val = '/home2/xuzihao/zzs/overlook_44w_likeCoco/val'
with open(main_txt, 'r') as fr:
    line = fr.readline().strip()
    count = 0
    while line:
        count += 1
        print(count)
        # if count == 50:
        #     break
        if line.endswith('.jpg'):
            if count%20 == 0:
                image_dir = output_dir_val
            else:
                image_dir = output_dir_train
            img_path = os.path.join(main_dir, line)
            dist_path = os.path.join(image_dir, 'images', line)
            shutil.copy(img_path, dist_path)
            txt_name = line.replace('.jpg', '.txt')
            txt_path = os.path.join(image_dir, 'labels', txt_name)
            with open(txt_path, 'w') as fw:
                line = fr.readline().strip()
                height = int(line.split(',')[1][1:-1])
                width = int(line.split(',')[0][1:])
                while line:     
                    line = fr.readline().strip()
                    if not line:
                        break
                    type = line[1]
                    if type == '0':
                        pass
                    else:
                        head_str = line[5:-2].split(', ')  
                        head_list = list(map(int, head_str))
                        head_center_x, head_center_y, head_w, head_h = xyxy2xywh(head_list[:4])
                        label = '0' + ' ' + str(round(head_center_x/width, 6)) + ' ' + str(round(head_center_y/height, 6))  \
                        + ' '   + str(round(head_w/width, 6)) + ' ' + str(round(head_h/height, 6))
                        fw.write(label)
                        fw.write('\n')
        line = fr.readline().strip()





