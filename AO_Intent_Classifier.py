from flask import Flask,render_template,session,url_for,redirect
import numpy as np
import pandas as pd
from flask_wtf import FlaskForm
from wtforms import TextField,SubmitField, TextAreaField, validators
from tensorflow.keras.models import load_model
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import string 
import re

def return_prediction(model, sample_json):

    
    df = pd.read_csv('intent1-corrected.csv', names = ['Text','Intent'], encoding='utf-8')
    
    tokenizer3 = Tokenizer(num_words=50000, split=' ') #create a token based on 

    tokenizer3.fit_on_texts(df['Text'])
    sequenced = tokenizer3.texts_to_sequences(df['Text'])
    s_len = sample_json['sepal_length']
   
    input_text = s_len.lower()

    print("\n You Entered this =  ", input_text)
    input_text = ''.join(re.sub("[^a-zA-Z-']"," ", input_text))

    input_text = ' '.join(word for word in input_text.split() if len(word)>=3)

    words = input_text.split()
    
    print("\n\n Your word is = ", words)

    print("\n\n")

    word_index = tokenizer3.word_index

    x_tests = [[word_index[word] if (word in word_index and word_index[word]<=30000) else 0 for word in words]]
                
    #tokenizer3 = Tokenizer(num_words=50000)
    #input_text= "jawar".lower()                                   
    #tokenizer3.fit_on_texts([input_text])
  
    seq_pad = pad_sequences(x_tests, maxlen=33)
                                             
    #seq_pad = np.array([seq_pad.flatten()])
                        
    class_ind = model.predict(seq_pad)

    classes = ['Wish','Suggestion',' Positive', 'Question',  'Suggestion', 'Negative']
	
    #print("\n It is Predicted as  = ", classes[class_ind][0][0])
    #print("\n It is Probality is  = ", proba)

    #result = classes[class_ind][0][0]#, proba
    #print("\n")
    print(class_ind, classes[np.argmax(class_ind)])
    proba = classes[np.argmax(class_ind)]
    result = proba, class_ind[0][0]

    return result

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'

class FlowerForm(FlaskForm):

	text_input = TextAreaField("Enter Your Text", [validators.Length(min=3, max=100000)], render_kw={"placeholder": "Maaloo iddoo kanatti Barreeffama keessan galchaa!..."})

	submit = SubmitField("Analyze")

@app.route("/",methods=['GET','POST'])
def index():

	form = FlowerForm()
	
	if form.validate_on_submit():

		session['text_input'] = form.text_input.data
		
		return redirect(url_for("prediction"))

	return render_template('home.html',form=form )
	
#model = tf.keras.models.load_model('Intent_Classification_using_LSTM.h5', custom_objects={'tf': tf})

new_model = keras.models.load_model('Intent_Classification.h5')

@app.route('/prediction')
def prediction():
	
	content = {}

	content['sepal_length'] = session['text_input']
	
	results, class_ind = return_prediction(new_model, content)

	return render_template('prediction.html',results=results, class_ind = class_ind)

if __name__=='__main__':
	app.run(debug=True)
