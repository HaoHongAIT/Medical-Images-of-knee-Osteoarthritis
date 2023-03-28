from flask import Flask
from flask import render_template, request, redirect, jsonify
import os, uuid, tensorflow
from api import Pipeline

app = Flask(__name__)
app.secret_key = '1234567890qwertyuiop'
app.config['UPLOAD_FOLDER'] = './static/images/'
# YANDEX_API_KEY = '1234567890qwertyuiop'
# app.wsgi_app = WhiteNoise(app.wsgi_app, root='static/')
model = tensorflow.keras.models.load_model("./model/weight/model_InceptionV3_DenseNet201_weights.h5")


@app.route('/<path:path>')
def static_file(path):
    return app.send_static_file(path)

# 
def allowed_file(filename):
    """check if file extension is right
    
    Keyword arguments:
    argument -- uploaded filename
    Return: boolean
    """    
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in set(['png', 'jpg', 'jpeg'])

# force browser to hold no cache. Otherwise old result returns.
# @app.after_request
# def set_response_headers(response):
#     response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
#     response.headers['Pragma'] = 'no-cache'
#     response.headers['Expires'] = '0'
#     return response

@app.route('/', methods=['GET', 'POST'])
def upload_file():    
    try:
        # remove files created more than 5 minute ago
        os.system("find static/images/ -maxdepth 1 -mmin +5 -type f -delete")
    except OSError:
        pass
    if request.method.__eq__('POST'):
        # check if the post request has the file part
        if 'img-file' not in request.files:
            print("don't have image file uploaded")
            return redirect(request.url)
        knee_xray = request.files['img-file']
        files = [knee_xray]        
        img_name = str(uuid.uuid4()) + ".png"
        file_names = [img_name]
        for i, file in enumerate(files):                    
            if file.filename == '':
                # if user does not select file, browser also
                return redirect(request.url)
            if file and allowed_file(file.filename):
                # submit an empty part without filename
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file_names[i]))        
        
        # pipeline        
        pipeline = Pipeline(img_name=file_names[-1])
        pipeline.detect_to_crop()
        pipeline.pre_processing()
        pipeline.predict(model)
        
        return render_template('success.html', **pipeline.result)
    return render_template('upload.html')

# @app.route('/api', methods=['GET', 'POST'])
# def api():
#     dict = None
#     return jsonify(dict)


@app.errorhandler(404)
def page_not_found(error):
    return render_template('page_not_found.html'), 404

if __name__ == '__main__':        
    app.run(debug=True)