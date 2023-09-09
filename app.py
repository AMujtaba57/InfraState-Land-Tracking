from flask import Flask, request, jsonify, send_file
import requests, os, cv2
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from utils.inference import detection
from utils.get_change_imgs import get_change_with_ssim
from werkzeug.utils import secure_filename

app = Flask(__name__)

load_dotenv()
API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')

zoom = 19
size = '640x480'

ALLOWED_EXTENSIONS = ('png', 'jpg', 'jpeg', 'webp')

satelite_img_folder = 'satelite_image/'
os.makedirs(satelite_img_folder, exist_ok=True)

result_img_folder = 'result_image/'
os.makedirs(result_img_folder, exist_ok=True)

UPLOAD_FOLDER = 'get_change_uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/get_satellite_image/', methods=['POST'])
def get_satellite_image():
    if request.method == 'POST':
        latitude = request.args['latitude']
        longitude = request.args['longitude']

        api_url = f"https://maps.googleapis.com/maps/api/staticmap?center={latitude},{longitude}&zoom={zoom}&size={size}&maptype=satellite&key={API_KEY}"

        response = requests.get(api_url)

        if response.status_code == 200:
            image_filename = "{:.3f}_{:.3f}_image.png".format(float(latitude), float(longitude))
            image_path = os.path.join(satelite_img_folder, image_filename)

            result_path = os.path.join(result_img_folder, image_filename)

            with open(image_path, "wb") as f:
                f.write(response.content)

            result = detection(image_path)

            vmin = 0
            vmax = 2

            plt.imshow(result[0].argmax(axis=2), vmin = vmin, vmax = vmax)
            plt.savefig(result_path)

            return send_file(result_path, mimetype='image/png')
        else:
            return jsonify({'error': 'Image download failed.'}), response.status_code
        
    else:
        return jsonify({'error': 'Invalid request format'}), response.status_code


@app.route('/get_change/', methods=['POST'])
def get_change_image():
    if request.method == 'POST':
        if request.files['current_img'] != "" and request.files['previous_img'] != "":
            current_image = request.files['current_img']
            previous_image = request.files['previous_img']

            valid_current_image = current_image.filename.lower().endswith(ALLOWED_EXTENSIONS)
            valid_previous_image = previous_image.filename.lower().endswith(ALLOWED_EXTENSIONS)

            if valid_current_image and valid_previous_image:
                current_filename = secure_filename(current_image.filename)
                previous_filename = secure_filename(previous_image.filename)

                file1_loc = os.path.join(UPLOAD_FOLDER, current_filename)
                file2_loc = os.path.join(UPLOAD_FOLDER, previous_filename)

                current_image.save(file1_loc)
                previous_image.save(file2_loc)

                current_img = cv2.imread(file1_loc)
                previous_img = cv2.imread(file2_loc)

                change_overlay, change_percentage = get_change_with_ssim(current_img, previous_img)
                print(f"Change Percentage: {change_percentage:.2f}%")

                result_path = os.path.join(result_img_folder, "track_image.jpg")
                cv2.imwrite(result_path, change_overlay)

                # return send_file(result_path, mimetype='image/jpeg')
                return jsonify({'change': f'{change_percentage: .3f}', 'Image-Path': f'{result_path}'}), 200

            else:
                return jsonify({'error': 'Unauthorized'}), 405

        else:
            return jsonify({'error': 'Image uploading failed.'}), 403
    else:
        return jsonify({'error': 'Invalid request format'}), 500
                    


if __name__ == '__main__':
    app.run(debug=False)
