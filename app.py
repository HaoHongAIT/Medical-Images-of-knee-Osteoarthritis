from flask import Flask
from flask import render_template, request, redirect
import os, uuid
from api import Pipeline


app = Flask(__name__)
app.secret_key = '1234567890qwertyuiop'
app.config['UPLOAD_FOLDER'] = './static/images/'
# YANDEX_API_KEY = '1234567890qwertyuiop'
# app.wsgi_app = WhiteNoise(app.wsgi_app, root='static/')
predict_img = Pipeline()

@app.route('/<path:path>')
def static_file(path):
    return app.send_static_file(path)

# check if file extension is right
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in set(['png', 'jpg', 'jpeg'])

# force browser to hold no cache. Otherwise old result returns.
# @app.after_request
# def set_response_headers(response):
#     response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
#     response.headers['Pragma'] = 'no-cache'
#     response.headers['Expires'] = '0'
#     return response

# main directory of programme
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
            # if user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                return redirect(request.url)
            if file and allowed_file(file.filename):
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file_names[i]))
        
        # pipeline
        predict_img.pre_processing(img_name=file_names[-1])
        result_params = predict_img.predict()
        
        return render_template('success.html', **result_params)
    return render_template('upload.html')


@app.errorhandler(404)
def page_not_found(error):
    return render_template('page_not_found.html'), 404

if __name__ == '__main__':        
    app.run(debug=True)