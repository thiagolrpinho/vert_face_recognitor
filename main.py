from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify, flash
import os
from werkzeug.utils import secure_filename
import sys
from image_helpers import open_crop_and_resize_face
from encoding_helpers import is_match, faces_to_embeddings

# Based on https://github.com/tylerfreckmann/cas-api
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
UPLOAD_FOLDER = './uploads'
app = Flask(__name__)
app.secret_key = "TEU_PAI"
app.config["CACHE_TYPE"] = "null"
app.app_context.
@app.route('/')
def index():
    return render_template('index.html', host=APP_IP, port=str(APP_PORT))

@app.route('/reconhecimento_facial/')
def reconhecimento_index():
    return render_template('reconhecimento_index.html')

          filename = "first_image.jpg"
          if i == 1:
            filename = "second_image.jpg"
          path = os.path.join(UPLOAD_FOLDER, filename)
          os.remove(path)
          image.save(path)
          paths.append(path)

@app.route('/reconhecimento_facial/result')
def reconhecimento_result(match_result):
    if match_result:
        return render_template('is_same.html')
    else:
        return render_template('not_same.html')

@app.route('/reconhecimento_facial/upload', methods=['GET', 'POST'])
def reconhecimento_upload():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'first_image' not in request.files or\
                'second_image' not in request.files:
            print('No file part')
            return redirect(request.url)
        images = []
        faces = []
        paths = []
        images.append(request.files['first_image'])
        images.append(request.files['second_image'])
        # if user does not select file, browser also
        # submit an empty part without filename
        for i, image in enumerate(images):
            if image.filename == '':
                print('No selected file')
                return redirect(request.url)
            if image and allowed_file(image.filename):
                ##### ERRO #####
                # Nessa etapa está sendo alterada a extensão do arquivo 
                # para por exemplo .jpg0 .jpg1 e etc
                # o arquivo está sendo identificado com codificação inválida
                # e por algum motivo o arquivo não está sendo salvo
                filename = secure_filename(str(image.filename) + str(i))
                path = os.path.join(UPLOAD_FOLDER, filename)
                image.save(path)
                paths.append(path)

        for path in paths:
            face_nparray, original_image = open_crop_and_resize_face(path)
            faces.append(face_nparray)
            os.remove(path)
        embeddings = faces_to_embeddings(faces)
        return reconhecimento_result(is_match(embeddings[0], embeddings[1]))
    return redirect(url_for('reconhecimento_index'))


@app.route('/renach/')
def renach_index():
    return render_template('renach_index.html')


@app.route('/renach/upload', methods=['GET', 'POST'])
def renach_upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        # if user does not select file, browser also
        # submit an empty part without filename
        renach_image = request.files['file']
        if renach_image.filename == '':
            print('No selected file')
            return redirect(request.url)
        if renach_image and allowed_file(renach_image.filename):
            texto_extraido = renach_extrai_textos(renach_image)
            return texto_extraido
    return redirect(url_for('renach_index'))

def renach_extrai_textos(image):
    return 'Belo conteúdo extraído da renach'

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    if len(sys.argv) == 6:
    # TODO - user upload astore 
    # To start app only want to use APP_IP, APP_PORT, UPLOAD_FOLDER params
        APP_IP = sys.argv[1]
        APP_PORT = int(sys.argv[2])
        UPLOAD_FOLDER = sys.argv[3]
        AUTHINFO = sys.argv[4]
        ASTORE = sys.argv[5]
        # ASTORE_LIB = sys.argv[6]
    else:
        APP_IP = '0.0.0.0'
        APP_PORT = 7050
        UPLOAD_FOLDER = './uploads'
        AUTHINFO = './.authinfo'
        ASTORE = 'lenet'
        # ASTORE_LIB = 'casuser'
    app.run(debug=True, host=APP_IP, port=APP_PORT)



