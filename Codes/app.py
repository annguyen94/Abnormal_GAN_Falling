from flask import Flask, request, render_template, redirect, url_for, send_from_directory
#from flask_ngrok import run_with_ngrok
from werkzeug.utils import secure_filename
import os
import cv2
import shutil
import time

def dir_last_updated(folder):
    return str(max(os.path.getmtime(os.path.join(root_path, f))
               for root_path, dirs, files in os.walk(folder)
               for f in files))

def make_overwritefolder(path):
    if os.path.exists(path):
        files = os.listdir(path)
        for file_name in files:
            file_path = path + '/' + file_name;
            os.remove(file_path)
    else:
        os.makedirs(path)

UPLOAD_FOLDER = '../uploads/01'
ALLOWED_EXTENSIONS = set(['jpg'])
frames_path = 'frames'
mask_path = 'mask'
plot_path = 'static/plot'
result_video_path = 'static/videos'


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER # upload file
#run_with_ngrok(app)  # Start ngrok when app is run

@app.route('/', methods=['GET', 'POST'])
def home():
    
    #print (request.method)  
    make_overwritefolder(UPLOAD_FOLDER)
    make_overwritefolder(plot_path)
    make_overwritefolder(frames_path)
    make_overwritefolder(result_video_path)
    make_overwritefolder(mask_path)
    if request.method == 'GET':
        
        return render_template('index.html')
        
        
    if request.method == 'POST':
        pretrain = request.form['pretrain_options']
        
        print ("pretrain", pretrain)
        dataset = request.form.get('data_options')

        print ("dataset", dataset)
        if dataset!=None:
            test_folder = '../Data/' + dataset + '/testing/frames'
        else:
            dataset = 'upload'
            test_folder = '../uploads'
            uploaded_files = request.files.getlist('file[]')
            
            for _file in uploaded_files:
                filename = secure_filename(_file.filename)
                _file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        video_name = dataset
        out_video = result_video_path + '/' + video_name + '.mp4'
        make_video_script = 'ffmpeg -f image2 -r 20 -i ./frames/%06d.jpg -vcodec libx264 -acodec aac -y ' + out_video
        
        run_ano_script = 'python app_inference.py --dataset ' + dataset +' --test_folder ' + test_folder + ' --gpu 0 --snapshot_dir checkpoints/pretrains/'+pretrain

        os.system(run_ano_script)
        os.system(make_video_script)
		
        list_plots = [ plot_path + '/' + i for i in os.listdir(plot_path)]
        print ("list plot:", list_plots)
        return render_template('result.html', video_path=out_video, list_plots=list_plots, dataset=dataset, last_updated=dir_last_updated('static'))
    
    #list_plots = [ plot_path + '/' + i for i in os.listdir(plot_path)]
    #return render_template('result.html', video_path='static/videos/ped2.mp4', list_plots=list_plots, dataset='ped2')
if __name__ == '__main__':
    app.run()
