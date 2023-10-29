import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Load your machine learning model
with open('model_save.sav', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        features = [[sepal_length, sepal_width, petal_length, petal_width]]
        prediction = model.predict(features)

        if prediction == 0:
            prediction_label = "Iris-setosa"
            image_path = 'static/images/Iris_setosa.jpg'  # Path to the image for Iris-setosa
        elif prediction == 1:
            prediction_label = "Iris-versicolor"
            image_path = 'static/images/Iris_versicolor.jpg'  # Path to the image for Iris-versicolor
        elif prediction == 2:
            prediction_label = "Iris-virginica"
            image_path = 'static/images/iris_virginica.jpg'  # Path to the image for Iris-virginica
        else:
            prediction_label = "Unknown"
            image_path = 'static/images/unknown.jpg'  # Default image for unknown predictions

        return render_template('index.html', prediction=prediction_label, image_path=image_path)
    except Exception as e:
        return render_template('index.html', prediction=f'Error: {e}')

if __name__ == '__main__':
    app.run(debug=True)


if __name__ == '__main__':
    app.run(debug=True)
