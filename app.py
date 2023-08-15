from flask import Flask, request, jsonify, send_file
import requests, os
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from utils.inference import detection

app = Flask(__name__)

load_dotenv()
API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')

zoom = 19
size = '640x480'

satelite_img_folder = 'satelite_image/'
os.makedirs(satelite_img_folder, exist_ok=True)

result_img_folder = 'result_image/'
os.makedirs(result_img_folder, exist_ok=True)

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


if __name__ == '__main__':
    app.run(debug=False)
