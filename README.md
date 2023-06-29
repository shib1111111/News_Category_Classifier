# News_Category_Classifier
News Classifier is a web application that uses machine learning techniques to classify news articles into different categories. It utilizes a trained model to predict the category of a given news article.
## File Structure
The file structure of the news_classifier application is as follows:
```
news_classifier
    └─────├── requirements.txt
          ├── News_Category_Dataset_v3.json
          ├── main.py
          ├── scraping_module.py
          ├── prediction_module.py
          ├── trained_model.h5
          ├── categories.npy
          ├── max_length.npy
          ├── tokenizer.pkl
          └───templates
          └───index.html
          └───result.html
```
## File Descriptions
**requirements.txt**: A text file listing the required Python packages and their versions for running the application.

**News_Category_Dataset_v3.json** : This dataset contains around 210k news headlines from 2012 to 2022 from HuffPost. This is one of the biggest news datasets and can serve as a benchmark for a variety of computational linguistic tasks. HuffPost stopped maintaining an extensive archive of news articles sometime after this dataset was first collected in 2018, so it is not possible to collect such a dataset in the present day. Due to changes in the website, there are about 200k headlines between 2012 and May 2018 and 10k headlines between May 2018 and 2022. [download link](https://drive.google.com/file/d/1ZIMwxLelFGsGAjgfvsSqOdq51QjmY97l/view?usp=sharing)
 - **Content**
      - Each record in the dataset consists of the following attributes:
         - _category_: category in which the article was published.
         - _headline_: the headline of the news article.
         - _authors_: list of authors who contributed to the article.
         - _link_: link to the original news article.
         - _short_description_: Abstract of the news article.
         - _date_: publication date of the article.

**main.py**: The main Python file that runs the web application and handles the routing and requests.

**scraping_module.py**: A Python module that handles web scraping functionality to retrieve news articles for classification.

**prediction_module.py**: A Python module that contains functions for preprocessing input and predicting the category using the trained model.

**trained_model.h5**: A trained machine learning model in HDF5 format, used for predicting the category of news articles. [download link](https://drive.google.com/file/d/14Bd5mNufOY3wW0JLENy3xeYSCxvLRzYF/view?usp=sharing)

**categories.npy**: A NumPy file containing the list of categories used for classifying news articles.

**max_length.npy**: A NumPy file containing the maximum length of input sequences used during training the model.

**tokenizer.pkl**: A Pickle file containing the tokenizer object used for tokenizing input text during preprocessing.

**templates**: A directory containing HTML templates used for rendering the web application.

## Getting Started
To run the News Classifier web application, follow these steps:

* Clone the repository or download the source code files.

* Download All other required files(.pkl, .npy, .h5 files) from [github link](https://github.com/shib1111111/News_Category_Classifier/edit/main/README.md)
* Navigate to the project directory: ```cd news_classifier```

* (Optional) Create a virtual environment: ```python3 -m venv venv``` (for Linux/macOS) or ```python -m venv venv``` (for Windows).

* Activate the virtual environment:

     * For Linux/macOS: ```source venv/bin/activate```

     * For Windows: ```venv\Scripts\activate.bat```

* Install the required packages: ```pip install -r requirements.txt```

* Run the application: ```python main.py```

Open a web browser and visit http://localhost:5000 to access the News Classifier application.

## Usage
The News Classifier web application allows you to classify news articles into different categories. Here's how to use it:

* Open the application in a web browser.

* Enter the URL of a news article or paste the content of the article in the provided text area.

* Click on the _**Classify**_ button to classify the news article.

* The predicted category will be displayed on the screen.


### requirements.txt
```
Flask
pandas
tensorflow
scikit-learn
beautifulsoup4
torch
numpy
```

## Algorithms and Logic:

### **main.py**
The main.py file in the News Classifier web application contains the main logic and routing for handling requests. Here's a breakdown of the algorithms used in the file:
* Import necessary modules and functions:
   - Import the Flask class from the flask module for creating the Flask application.
   - Import functions from other modules (predict and load_model_components) to handle prediction and loading model components.
   - Import the _scrape_content_ function from the scraping_module to retrieve the content of a news article.
* Create the Flask application:
   - Initialize a Flask application using ```Flask(__name__)```
* Define the home route:
   - Define a route using the ```@app.route('/')``` decorator.
   - Create a home function that renders the index.html template.
   - Return the rendered template.
* Define the prediction route:
   - Define a route using the @app.route('/predict', methods=['POST']) decorator. This route handles the form submission for news article prediction.
   - Create a make_prediction function that:
   - Retrieves the URL of the news article from the form using request.form['url'].
   - Calls the scrape_content function to get the headline and description of the news article.
   - Calls the load_model_components function to load the tokenizer, maximum sequence length, categories, and the model.
   - Calls the predict function with the loaded components, headline, description, and categories to get the predicted class.
   - Renders the result.html template, passing the URL, headline, description, and predicted class as variables.
         - Return the rendered template.
* Run the application:
   - Add the if ```__name__ == __main__``` block to ensure that the Flask application is only run when the script is executed directly, not when imported as a module.
   - Inside the block, run the application using ```app.run(debug=True)```

### **scraping_module.py**
The scrape_content function from _scraping_module.py_ file utilizes the BeautifulSoup library and requests module to scrape the content of a news article from a given URL. Here's the algorithm for the scrape_content function:
* Import necessary modules:
  - Import the BeautifulSoup class from the bs4 module for parsing HTML content.
  - Import the requests module for making HTTP requests.
* Define the scrape_content function that takes the url parameter.
* Send an HTTP GET request to the specified URL using ```requests.get(url)```. Assign the response to the response variable.
* Create a BeautifulSoup object by passing the ```response.content``` and the parser type, ```html.parser``` to the BeautifulSoup constructor. Assign the object to the soup variable.
* Extract the headline of the news article:
    - Find the HTML element that contains the headline using ```soup.find('h1')```
    - Retrieve the text content of the headline element using ```.get_text()```
    - Assign the headline to the headline variable.
* Extract the description of the news article:
    - Find all the HTML elements that contain the description using ```soup.find_all('p')```
    - Iterate over the description elements and extract their text content using ```.get_text()```
    - Join the text content of all description elements into a single string using ```''.join([tag.get_text() for tag in description])```
    - Assign the description text to the description_text variable.
    - If the length of the description text is 0, it means no description was found. In this case, assign the headline as the description text.
* Return the headline and description_text as a tuple.


### **prediction_module.py**
The **_prediction_module.py_** file in the News Classifier application contains functions for training a model and making predictions on news articles. It utilizes various libraries and techniques to preprocess the data, build a machine-learning model, and achieve high accuracy on the training dataset.

* **Training the Model**
     - The train_model function is responsible for training the machine learning model. Here's the algorithm for the train_model function:
     - Read the news data from a JSON file into a list.
     - Create a data frame from the list of news data and select relevant columns (link, headline, short_description, category).
     - Initialize a tokenizer to tokenize the headlines and short descriptions.
     - Fit the tokenizer on the combined text of headlines and short descriptions.
     - Convert the tokenized sequences into padded sequences of equal length.
     - Create input features (X) from the padded sequences.
     - Encode the target categories (y) using one-hot encoding.
     - Determine the number of classes by retrieving the shape of y.
     - Obtain the unique list of categories from the DataFrame.
     - Split the data into training and validation sets using a specified test size and random state.
     - Build a sequential model using TensorFlow/Keras with an embedding layer, flatten layer, and dense layer.
     - Compile the model with categorical cross-entropy loss and the Adam optimizer.
     - Train the model on the training data for a specified number of epochs and batch size.
     - Save the trained model, tokenizer, maximum sequence length, and categories to separate files.
* **Loading Model Components**
     - The load_model_components function loads the saved tokenizer, maximum sequence length, categories, and the trained model. Here's the algorithm for the load_model_components function:
     - Initialize variables for tokenizer, maximum sequence length, categories, and model.
     - Attempt to load the tokenizer from the pickle file.
     - Load the maximum sequence length and categories from their respective NumPy files.
     - Load the trained model from the HDF5 file.
     - If any of the files are not found or an error occurs, appropriate error messages are displayed.
     - Return the loaded tokenizer, maximum sequence length, categories, and model.
* **Making Predictions**
     - The predict function takes the tokenizer, maximum sequence length, model, headline, description, and categories as inputs. Here's the algorithm for the predict function:
     - Check if the tokenizer is initialized. If not, raise a ValueError.
     - Tokenize the headline and description using the tokenizer.
     - Pad the tokenized sequence to match the maximum sequence length.
     - Make a prediction using the model by calling ```model.predict```
     - Find the index with the highest predicted probability using ```np.argmax```
     - Retrieve the predicted class from the categories list using the predicted class index.
     - Return the predicted class.
* **Benefits of the Model**
The model trained using the News Classifier achieves an impressive accuracy of _**99.65%**_ on the training dataset. Here are some benefits of using this model:
  - **High Accuracy** :The model demonstrates a high level of accuracy on the training data, indicating its ability to correctly classify news articles into different categories.
  - **Efficient Text Representation** : The model utilizes tokenization and embedding techniques to represent news article text in a numerical format suitable for machine learning. This allows the model to capture semantic information and make accurate predictions.
  - **Deep Learning Architecture** : The model employs a sequential architecture with an embedding layer, flatten layer, and dense layer. This deep learning architecture enables the model to capture complex patterns and relationships in the news article data, leading to improved prediction accuracy.
  - **Transfer Learning** : The model utilizes pre-trained transformer-based models like BERT for sequence classification. This leverages the knowledge learned from large-scale pre-training tasks, resulting in better performance and faster convergence during training.
  - **Efficient Training Process** : The model is trained using TensorFlow/Keras, which provides a user-friendly and efficient interface for deep learning. The training process involves splitting the data into training and validation sets, compiling the model with appropriate loss and optimizer, and iteratively updating the model's parameters to minimize the loss.
  - **Serialization and Reusability**: The model components such as the tokenizer, maximum sequence length, categories, and the trained model are serialized and saved to files. This allows for easy reusability and deployment of the trained model, as it can be loaded and used independently without retraining.
  - **Flexible Prediction**: The model's prediction function accepts the headline, description, tokenizer, maximum sequence length, model, and categories as inputs. It handles the necessary preprocessing steps and returns the predicted class, enabling seamless integration with the web application.

### **templates**

```
└───templates
          └───index.html
          └───result.html
```
The templates directory contains two HTML files: index.html and result.html. These templates are used to render the home page and the result page of the web application, respectively.

### **License**
This project is licensed under the MIT License.


