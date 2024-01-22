import tensorflow as tf
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image

st.set_page_config(
    page_title="DigitRecognizer",
    page_icon="💬",
    layout="wide",
)

if 'model' not in st.session_state:
    st.session_state['model'] = tf.keras.models.load_model('model3')


def center_image(img):

    # Convert to grayscale and binarize
    img = img.convert('L')
    img_array = np.array(img)
    img_array = (img_array > 0).astype(int)  # Assuming white digit on black background

    if np.all(img_array == 0):
        return img

    # Binarize the image

    # Calculate center of mass
    coordinates = np.argwhere(img_array)
    x_mean = int(coordinates[:, 0].mean())
    y_mean = int(coordinates[:, 1].mean())

    # Get image dimensions
    height, width = img_array.shape

    # Calculate shift required
    x_shift = np.round(width / 2 - y_mean).astype(int)
    y_shift = np.round(height / 2 - x_mean).astype(int)

    # Create a new blank image and paste the shifted digit
    new_img = Image.new("L", (width, height), color=0)
    new_img.paste(img, (x_shift, y_shift))

    return new_img

# Usage example
# original_img = Image.open("path_to_your_image.jpg")
# centered_img = center_image(original_img)
# centered_img.show()


def preprocess(image_data):
    image = Image.fromarray((image_data.astype('uint8')), mode='RGBA')
    image = center_image(image)
    image = image.convert("L")
    image = image.resize((28, 28))
    # image = image.filter(ImageFilter.EDGE_ENHANCE)
    # image = image.resize((28, 28))
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)

    return image_array


def predict(image):
    return np.argmax(st.session_state['model'].predict(image))


st.title("Digit Recognizer")
st.subheader("")
st.caption("***Note: If you're using a mobile device, please scroll down to see the prediction result after drawing.***")
st.markdown("Draw a digit in the box below :")


col0, col1 = st.columns(2)
with col0:
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=30,
        stroke_color="#fefefe",
        background_color="#000000",
        width=400,
        height=400,
        drawing_mode="freedraw",
        update_streamlit=True,
        key="canvas")

img = np.zeros(shape = (1,28,28,1))
if canvas_result.image_data is not None:
    image_data = canvas_result.image_data
    img = preprocess(image_data)


# st.image(img)
with col1:
    st.title(f'***Your Drawing Resembles*** {"Nothing" if np.all(img == 0) else predict(img)}')
    # st.title("None" if np.all(img == 0) else predict(img))

st.caption("***Tips: Use the Dustbin icon to clear canvas, and Arrow buttons for undo/redo.***")

# if st.button('Predict'):
    # Here you would add the code to process the canvas image and make a prediction
    # For example:
    # image = process_canvas(canvas_result.image_data)

# mnist = tf.keras.datasets.mnist
# (X_train , Y_train) , (X_test , Y_test) = mnist.load_data()
# X_train , X_test = X_train/255.0 , X_test/255.0
# i = 16
# image = X_test[i]
# st.image(image)

    # prediction = model_predict(image)
    # st.write('Predicted Digit:', prediction)


