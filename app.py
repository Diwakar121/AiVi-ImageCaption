from flask import Flask, render_template, url_for, request, redirect
from caption import *
import warnings
import os
warnings.filterwarnings("ignore")



app = Flask(__name__)


@app.route('/')
def hello():
    return render_template('index.html')


@app.route('/', methods = ['POST'])
def upload_file():
	if request.method == 'POST':
		img = request.files['image']

		# print(img)
		# print(img.filename)

		img.save("static/"+img.filename)

	
		caption = caption_this_image("static/"+img.filename)



		
		result_dic = {
			'image' : "static/" + img.filename,
			'description' : caption
		}
	return render_template('index.html', results = result_dic)


@app.route('/findcaption', methods = ['POST'])
def getcaption():
	data = request.data
    # print(data)
		# img.save("static/"+img.filename)
        # caption = caption_this_image("static/"+img.filename)
        
		# result_dic = {
		# 	'image' : "static/" + img.filename,
		# 	'description' : caption
		# }
	d={"working":"checked"}
	return jsonify(d);

 

port = int(os.environ.get("PORT", 5000))
app.run(host='0.0.0.0', port=port)