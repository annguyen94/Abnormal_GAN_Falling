import tensorflow as tf
import os
import time
import numpy as np
import pickle
import cv2
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from models import generator
from utils import DataLoader, load, save, psnr_error
from constant import const
import app_evaluate as evaluate


slim = tf.contrib.slim

os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = const.GPU

dataset_name = const.DATASET
test_folder = const.TEST_FOLDER

num_his = const.NUM_HIS
height, width = 256, 256

snapshot_dir = const.SNAPSHOT_DIR
psnr_dir = const.PSNR_DIR
evaluate_name = const.EVALUATE

print("This is const = ", const)

def image2_bin(img):
    #new_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    new_image = img
    new_image[new_image<30] = 0

    avg_val = np.mean(new_image[new_image!=0])
    
    new_image[new_image < avg_val] = 0
    '''    
    Sum = 0
    count = 0
    for im_col in new_image:
        for im_cel in im_col:
            if im_cel!=0:
                Sum += im_cel
                count+=1
    if count==0:
        return new_image
    avg_val = Sum//count
    new_image[new_image<avg_val]=0
    '''
    return new_image

def apply_mask(img, mask):
    h, w = img.shape[:2]

    #img = img.flatten()
    #mask = mask.flatten()
    
    
    yellow = np.zeros((h, w, 3), dtype='uint8')
    yellow[..., 0] = 0
    yellow[..., 1] = 255
    yellow[..., 2] = 255

    mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)

    
    mask = mask/255    

    img = (1-mask) * img + mask * yellow
    #img = mask

    '''
    for i in range(h):
        for j in range(w):
            if (mask[i][j] != 0):
                alpha = np.float32 ( mask[i][j]/255 )
                new_color = [(1-alpha) * img[i][j][0] + alpha * 0, (1-alpha) * img[i][j][1] + alpha * 255, (1-alpha) * img[i][j][2] + alpha * 255]
                img[i][j] = np.uint8 ( new_color )
    '''
    
    #img = (1-mask)*img + mask*np.uint8(0, 255, 255)
    
    #img = img.reshape(h, w, 3)
    return img

# define dataset
with tf.name_scope('dataset'):
    test_video_clips_tensor = tf.placeholder(shape=[1, height, width, 3 * (num_his + 1)],
                                             dtype=tf.float32)
    test_inputs = test_video_clips_tensor[..., 0:num_his*3]
    test_inputs = tf.reshape(test_inputs,[1,height,width,num_his,3])
    test_inputs = test_inputs[:,:,:,::2,:]
    test_inputs = tf.reshape(test_inputs,[1,height,width,-1])
    test_gt = test_video_clips_tensor[..., -3:]
    print('test inputs = {}'.format(test_inputs))
    print('test prediction gt = {}'.format(test_gt))

# define testing generator function and
# in testing, only generator networks, there is no discriminator networks and flownet.
with tf.variable_scope('generator', reuse=None):
    print('testing = {}'.format(tf.get_variable_scope().name))
    test_outputs = generator(test_inputs, layers=5, output_channel=3)
    print ("test_outputs: ",test_outputs)
    test_psnr_error = psnr_error(gen_frames=test_outputs, gt_frames=test_gt)
    truth = test_gt*255
    loss_val = tf.reduce_mean(tf.abs((test_outputs - test_gt)*255), axis=3, keep_dims=True)
    #loss_val[loss_val<30] = 0
    #mean = tf.reduce_mean(loss_val)
    #loss_val[loss_val<mean] = 0
    psnr_and_mask = test_psnr_error, loss_val, truth


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    # dataset
    data_loader = DataLoader(test_folder, height, width)

    # initialize weights
    sess.run(tf.global_variables_initializer())
    print('Init global successfully!')

    # tf saver
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=None)

    restore_var = [v for v in tf.global_variables()]
    loader = tf.train.Saver(var_list=restore_var)

    def inference_func(ckpt, dataset_name, evaluate_name):
        load(loader, sess, ckpt)

        psnr_records = []
        videos_info = data_loader.videos
        num_videos = len(videos_info.keys())
        total = 0
        timestamp = time.time()

        
        for video_name, video in videos_info.items():
            length = video['length']
            total += length
            psnrs = np.empty(shape=(length,), dtype=np.float32)

            for i in range(num_his, length):
                video_clip = data_loader.get_video_clips(video_name, i - num_his, i + 1)
                psnr = sess.run(test_psnr_error,
                                feed_dict={test_video_clips_tensor: video_clip[np.newaxis, ...]})
                psnrs[i] = psnr

                print('video = {} / {}, i = {} / {}'.format(video_name, num_videos, i, length))

            psnrs[0:num_his] = psnrs[num_his]
            psnr_records.append(psnrs)
        
        scores = np.array([], dtype=np.float32)
        # video normalization
        for i in range(num_videos):
            distance = psnr_records[i]

            distance -= distance.min()  # distances = (distance - min) / (max - min)
            distance /= distance.max()
            # distance = 1 - distance
            scores = np.concatenate((scores, distance[4:]), axis=0)

        used_time = time.time() - timestamp
        print('total time = {}, fps = {}'.format(used_time, total / used_time))
       
        inp_path = test_folder
        out_path = './frames/'
        
        it = 0
        testPredict = np.zeros(scores.shape, dtype=int)
        thres = 0.6
        
        for video_name, video in videos_info.items():
            
            length = video['length']
            new_video_name = video_name.split('\\')[-1]
            
            #save_npy_file = 'npy/' + video_name + '.npy'
            dat = np.zeros(length)
            
            frames_list = os.listdir(inp_path + '/' + new_video_name)
            frames_list.sort()
            for i in range(num_his, length):
                if scores[it] >= thres:
                    k=0
                else:
                    k=1
                testPredict[it] = k
                print('videos = {} / {}, i = {} / {}, Scores = {:.6f}, -{} - {} '.format(
                    video_name, num_videos, i, length, scores[it], 'Abnorm'if k==1 else 'Normal' , k))
                
                dat[i] = scores[it]
                
                # Make output video
                img_path = inp_path + '/' + new_video_name + '/' + frames_list[i]
                print("new_video_name", new_video_name)
                print ("img path:", img_path)
                frame_out = '/content/drive/MyDrive/SaverURFalling2Frame/frames/' + str(i) + '.jpg'
                print ("frames out:", frame_out)
                frame = cv2.imread(img_path)
                H, W = frame.shape[:2]
                
                video_clip = data_loader.get_video_clips(video_name, i - num_his, i + 1)
                l_val = sess.run(loss_val, feed_dict={test_video_clips_tensor: video_clip[np.newaxis, ...]})
                l_val = np.uint8(l_val)                
                l_val = l_val.reshape(256, 256, 1)                
                l_val = cv2.resize(l_val, (W,H))
                l_val = image2_bin(l_val)
                
                if k==1:
                    cv2.rectangle(frame, (0,0), (W, H), (0, 0, 255), thickness=5, lineType=8, shift=0)
                    #frame[l_val!=0] = (0,255,255)
                    # frame = apply_mask(frame, l_val)
                
                cv2.imwrite(frame_out, frame)
                
                it = it+1
                
            fig = plt.figure()
            lw = 2
            plt.plot(np.arange(4, length), dat[4:], color='darkorange', lw=lw)
            plt.xlim([0, length])
            plt.ylim([0.00, 1.00])
            plt.xlabel('Frame')
            plt.ylabel('Score')
            plt.title('Video '+ new_video_name +' Score')
            fig.savefig('./static/plot/'+ new_video_name + '.png')    
            #np.save(save_npy_file, dat)

    def label_inference_func(ckpt, dataset_name, evaluate_name):
        load(loader, sess, ckpt)

        psnr_records = []
        videos_info = data_loader.videos
        num_videos = len(videos_info.keys())
        total = 0
        timestamp = time.time()

        #mask_path = 'mask/'
        mask_list = []
        truth_list = []
        
        it = 0
        for video_name, video in videos_info.items():
            length = video['length']
            total += length
            psnrs = np.empty(shape=(length,), dtype=np.float32)

            for i in range(num_his, length):
                video_clip = data_loader.get_video_clips(video_name, i - num_his, i + 1)
                psnr, mask, truth = sess.run(psnr_and_mask,
                                feed_dict={test_video_clips_tensor: video_clip[np.newaxis, ...]})
                psnrs[i] = psnr
                
                mask = np.uint8(mask)               
                mask = image2_bin(mask)
                mask = mask.reshape(256, 256, 1)
                mask_list.append(mask)
                #np.save(mask_path + '{:06}'.format(it) + ".npy", mask)
                truth = np.uint8(truth)
                truth = truth.reshape(256, 256, 3)
                truth_list.append(truth)

                print('video = {} / {}, i = {} / {}'.format(video_name, num_videos, i, length))
                    
                it += 1

            psnrs[0:num_his] = psnrs[num_his]
            psnr_records.append(psnrs)

        end_time1 = time.time()
        used_time = end_time1 - timestamp
        print ("stamp1:", timestamp, "- end_time1", end_time1)
        print('total time = {}, fps = {}'.format(used_time, total / used_time))
        result_dict = {'dataset': dataset_name, 'psnr': psnr_records, 'flow': [], 'names': [], 'diff_mask': []}
        # TODO specify what's the actual name of ckpt.
        pickle_path = os.path.join(psnr_dir, os.path.split(ckpt)[-1])
        with open(pickle_path, 'wb') as writer:
            pickle.dump(result_dict, writer, pickle.HIGHEST_PROTOCOL)

        results, scores, labels, thres = evaluate.evaluate(evaluate_name, pickle_path)
        print(results)
        #end TODO
       
        inp_path = test_folder
        out_path = './frames/'
        
        it = 0
        testPredict = np.zeros(scores.shape, dtype=int)
        #thres = 0.6
        #mask_list = os.listdir(mask_path)
        #mask_list.sort()

        frames_list = os.listdir(inp_path + '/01')
        frames_list.sort()
        sample_img = cv2.imread(inp_path + '/' + '01' + '/' + frames_list[0])
        H, W = sample_img.shape[:2]

        timestamp2 = time.time()
        for video_name, video in videos_info.items():
            
            length = video['length']
            new_video_name = video_name.split('\\')[-1]
            
            #save_npy_file = 'npy/' + new_video_name + '.npy'
            dat = np.zeros(length)
            

            frames_list = os.listdir(inp_path + '/' + new_video_name)
            frames_list.sort()
            for i in range(num_his, length):
                if scores[it] >= thres:
                    k=0
                else:
                    k=1
                testPredict[it] = k
                print('videos = {} / {}, i = {} / {}, Scores = {:.6f}, -{} - {} '.format(video_name, num_videos, i, length, scores[it], 'Abnorm'if k==1 else 'Normal' , k))
                
                dat[i] = scores[it]
                
                # Make output video
                img_path = inp_path + '/' + new_video_name + '/' + frames_list[i]
                #print ("img path:", img_path)
                frame_out = "/content/drive/MyDrive/SaverURFalling2Frame/frames/" + str(i) + ".jpg"
                print ("frames out:", frame_out)
                frame = cv2.imread(img_path)
                H, W = frame.shape[:2]
                #frame = truth_list[it]
                #frame = cv2.resize(frame, (W, H))
                
                #l_val = np.load(mask_path + mask_list[it])               
                l_val = mask_list[it]
                l_val = cv2.resize(l_val, (W,H))
                #l_val = image2_bin(l_val)
                
                if k==1:
                    cv2.rectangle(frame, (0,0), (W, H), (0, 0, 255), thickness=5, lineType=8, shift=0)
                    #frame[l_val!=0] = l_val
                    # frame = apply_mask(frame, l_val)
                cv2.imwrite(frame_out, frame)
                
                it = it+1
            
            #print ("dat_shape", dat.shape)
            fig = plt.figure()
            lw = 2
            plt.plot(np.arange(4, length), dat[4:], color='green', lw=lw)
            plt.xlim([0, length])
            plt.ylim([0.00, 1.00])
            plt.xlabel('Frame')
            plt.ylabel('Score')
            plt.title('Video '+ new_video_name +' Score')
            fig.savefig('./static/plot/'+ new_video_name + '.png')
            #np.save(save_npy_file, dat)
        end_time2 = time.time()
        used_time2 = end_time2 - timestamp2
        print ("stamp2:", timestamp2, "- end_time2:", end_time2)
        print('total time = {}, fps = {}'.format(used_time2, total / used_time2))
    if dataset_name=='upload':
        inference_func(snapshot_dir, dataset_name, evaluate_name)
    else:
        label_inference_func(snapshot_dir, dataset_name, evaluate_name)
