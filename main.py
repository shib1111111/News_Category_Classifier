from flask import Flask, render_template, request
from prediction_module import predict,load_model_components
from scraping_module import scrape_content

app = Flask(__name__)
tokenizer = None
max_length = None


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def make_prediction():
    url = request.form['url']
    headline, description = scrape_content(url)
    tokenizer, max_length, categories, model = load_model_components()
    result = predict(tokenizer, max_length, model, headline, description, categories)
    return render_template('result.html',url =url, headline=headline, description=description, predicted_class=result)

if __name__ == '__main__':
    app.run(debug=True)
