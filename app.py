from flask import Flask, request, jsonify, render_template
from io import BytesIO
from PIL import Image
import base64
# import pickle
import pandas as pd
import numpy as np
# from mediapipe.python.solutions import (
#     holistic as mp_holistic,
#     drawing_utils as mp_drawing,
# )

app = Flask(__name__)

# Load the machine learning model and configure MediaPipe as before
# with open("model/body_language.pkl", "rb") as f:
#     model = pickle.load(f)

# drawing_specs = [
#     mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
#     mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
# ]


def process_frame(frame_data):
    try:
        # Decode the Blob data into an image
        image = Image.open(BytesIO(frame_data))

        # Convert the image to RGB
        image = image.convert("RGB")
        image_np = np.array(image)

        # Process the frame with MediaPipe Holistic
        # results = holistic.process(image_np)

        # Draw landmarks and perform body language analysis
        # if results.pose_landmarks:
        #     mp_drawing.draw_landmarks(
        #         image_np,
        #         results.pose_landmarks,
        #         mp_holistic.POSE_CONNECTIONS,
        #         *drawing_specs,
        #     )
        #     try:
        #         pose = results.pose_landmarks.landmark
        #         pose_row = list(
        #             np.array(
        #                 [
        #                     [landmark.x, landmark.y, landmark.z, landmark.visibility]
        #                     for landmark in pose
        #                 ]
        #             ).flatten()
        #         )
        #         X = pd.DataFrame([pose_row])
        #         body_language_class = model.predict(X)[0]
        #         body_language_prob = model.predict_proba(X)[0]
        #         if body_language_prob[np.argmax(body_language_prob)] > 0.98:
        #             # Process the image to draw text and rectangles as before
        #             pass
        #     except Exception as e:
        #         print(f"Error processing frame: {str(e)}")
        #         return {"error": "Frame processing error"}

        # Convert the processed frame back to base64 data URL
        buffer = BytesIO()
        Image.fromarray(image_np).save(buffer, format="JPEG")
        processed_frame_data_url = (
            "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
        )
        return {"processed_frame_data": processed_frame_data_url}

    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return {"error": "Frame processing error"}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process_frame", methods=["POST"])
def process_frame_route():
    try:
        # Read the Blob data from the request
        frame_data = request.get_data()

        # Process the frame and send the response
        result = process_frame(frame_data)
        return jsonify(result)
    except Exception as e:
        # Handle the error gracefully, e.g., return an error responses
        return jsonify({"error": "An error occurred while processing the frame."})


if __name__ == "__main__":
    app.run(debug=False)
