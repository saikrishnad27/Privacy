# Flask Packages
from flask import Flask,render_template,request,url_for
import requests
from bs4 import BeautifulSoup
import openai
import matplotlib.pyplot as plt
import nltk

import spacy
from nltk.corpus import stopwords
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import gensim
from gensim import corpora
from gensim.models import LdaMulticore
from gensim.models import CoherenceModel
from nltk.corpus import stopwords
# import certifi
# print(certifi.where())
# import os
# os.environ['SSL_CERT_FILE'] = certifi.where()
# os.environ['SSL_CERT_DIR'] = certifi.where()
# import ssl

# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context




# EDA Packages
import numpy as np 

## LDA packages
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from gensim.models import CoherenceModel
np.random.seed(2018)
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
import matplotlib.pyplot as plt
from gensim import corpora, models


app = Flask(__name__)
openai.api_key = "Put your key here"
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])



@app.route('/')
def index():
	image_url = url_for('static', filename='myimage.jpg')
	return render_template('index.html',image_url=image_url)

# Route for our Processing and Details Page
@app.route('/quiz',methods=['GET','POST'])
def dataupload():
	if request.method == 'POST':
		
		url = request.form['url']
		print(url)
		print("krishna")
		# Send a GET request to the URL
		response = requests.get(url)
		# Use BeautifulSoup to parse the HTML content
		soup = BeautifulSoup(response.content, "html.parser")
		#print(soup)
		privacy_keywords=["personal information", "data", "information", "cookies", "IP address",
                  "third party", "consent", "collection", "processing", "usage", "sharing",
                  "disclosure", "storage", "retention", "security", "access", "modification", 
                  "deletion", "account", "marketing", "opt-out", "opt-in", "user", "consumer",
                  "consent", "child", "age", "email", "address", "telephone", "credit card", "payment", "billing", 
                  "geolocation", "device", "browser", "session", "logs", "analytics", "tracking", "advertising", 
                  "cookies", "pixel", "server", "web beacon", "social media", "plugin", "profile", "data subject", 
                  "GDPR", "CCPA", "privacy shield", "transfer", "consent", "legitimate interest", "contract", "legal obligation", 
                  "vital interest", "public interest", "right to be forgotten", "right to access", "right to rectification", "right to erasure", "right to restrict processing", "right to data portability", "right to object", "data protection officer", "breach notification", "enforcement", "accountability", "privacy policy", "terms of service", "cookie policy", "user agreement", "privacy notice", "data controller", "data processor", "EU-U.S. Privacy Shield", "Safe Harbor", "Privacy Shield", "BCRs", "model clauses", "consent management", "cookie consent", "privacy settings", "privacy dashboard", "opt-out mechanism", "privacy seal", "trustmark", "information commissioner's office", "data protection authority",
                  "privacy rights", "data protection", "pseudonymization", "anonymization","name",'hipaa',     
                    'pipeda', 'coppa',"Demographic Data"]
		privacy_paragraphs = []
		#print("i")
		for paragraph in soup.find_all("p"):
			text = paragraph.get_text()
			if any(keyword in text.lower() for keyword in privacy_keywords):
				privacy_paragraphs.append(text)
		#print(privacy_paragraphs)
		sentences=[]
		nltk.download('punkt')
		for i in privacy_paragraphs:
			sentence = nltk.sent_tokenize(i)
			for j in sentence:
				sentences.append(j)
		tokens = []
		#nltk.download('stopwords')
		for sentence in sentences:
			sentence=sentence.lower()
			tokens.append(nltk.word_tokenize(sentence))
		stop_words = set(stopwords.words('english'))
		tokens_filtered=[]
		for i in tokens:
			stopped_tokens = [j for j in i if not j in stop_words]
			tokens_filtered.append(stopped_tokens)
		def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
			"""https://spacy.io/api/annotation"""
			texts_out = []
			for sent in texts:
				doc = nlp(" ".join(sent)) 
				texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
			return texts_out
		clean_text = lemmatization(tokens_filtered, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
		def prepare_corpus(doc_clean):
			"""
			Input  : clean document
			Purpose: create term dictionary of our courpus and Converting list of documents (corpus) into Document Term Matrix
			Output : term dictionary and Document Term Matrix
			"""
			# Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)
			dictionary = corpora.Dictionary(doc_clean)
			# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
			doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
			# generate LDA model
			return dictionary,doc_term_matrix
		dictionary,doc_term_matrix=prepare_corpus(clean_text)
		# def compute_coherence_values(dictionary, doc_term_matrix, doc_clean, stop, start, step):
		# 	"""
		# 	Input   : dictionary : Gensim dictionary
		# 			corpus : Gensim corpus
		# 			texts : List of input texts
		# 			stop : Max num of topics
		# 	purpose : Compute c_v coherence for various number of topics
		# 	Output  : model_list : List of LSA topic models
		# 			coherence_values : Coherence values corresponding to the LDA model with respective number of topics
		# 	"""
		# 	coherence_values = []
		# 	model_list = []
		# 	print("LDA MODEL")
		# 	for num_topics in range(start, stop, step):
		# 		# generate LdA model
		# 		print(num_topics)
		# 		model = LdaMulticore(doc_term_matrix, num_topics=num_topics, id2word = dictionary)  # train model
		# 		model_list.append(model)
		# 		print("Error1*****************")
		# 		coherencemodel = CoherenceModel(model=model, texts=doc_clean, dictionary=dictionary, coherence='c_v')
		# 		print("Error2*****************")
		# 		coherence_values.append(coherencemodel.get_coherence())
		# 	return model_list, coherence_values
		# def plot_graph(dictionary,doc_term_matrix,doc_clean,start, stop, step):
		# 	model_list, coherence_values = compute_coherence_values(dictionary, doc_term_matrix,doc_clean,
		# 															stop, start, step)
		# 	# Show graph
		# 	x = range(start, stop, step)
		# 	# plt.plot(x, coherence_values)
		# 	# plt.xlabel("Number of Topics")
		# 	# plt.ylabel("Coherence score")
		# 	# plt.legend(("coherence_values"), loc='best')
		# 	# plt.show()
		# 	temp=0
		# 	temp1=0
		# 	for m, cv in zip(x, coherence_values):
		# 		#print("Num Topics =", m, "has Coherence Value of", round(cv, 4))
		# 		if m==2:
		# 			temp=m
		# 			temp1=round(cv, 4)
		# 		else:
		# 			m1=round(cv, 4)
		# 			if temp1<m1:
		# 				temp1=m1
		# 				temp=m
		# 	return temp
		# start,stop,step=2,11,1
		# m=plot_graph(dictionary,doc_term_matrix,clean_text,start,stop,step)
		m=10
		lda_model = gensim.models.ldamodel.LdaModel(doc_term_matrix, id2word=dictionary, num_topics=m)
		def filter_topics(lda_model, fixed_keywords, num_topics):
			# Create a dictionary to hold the filtered topics
			filtered_topics = {}
			
			# Loop over each topic in the LDA model
			for topic_id in range(num_topics):
				# Get the topic words and probabilities
				words = lda_model.show_topic(topic_id, topn=10)
				# Check if the topic contains at least two of the fixed keywords
				keywords_found = 0
				for word, prob in words:
					if word in fixed_keywords:
						keywords_found += 1
					if keywords_found >= 2:
						filtered_topics[topic_id] = words
						break
						
			return filtered_topics
		filtered_topics = filter_topics(lda_model,privacy_keywords,m)
		check_list=[]
		for topic_id, words in filtered_topics.items():
			for word,prob in words:
				if word not in check_list:
					check_list.append(word)
		scores = []
		for paragraph in privacy_paragraphs:
			score = 0
			for keyword in check_list:
				if keyword in paragraph.lower():
					if keyword in privacy_keywords:
						score= score+(paragraph.count(keyword)*2)
					else:
						score=score+1
			score += len(nltk.sent_tokenize(paragraph))
			scores.append(score)
		top_paragraphs = [x for _,x in sorted(zip(scores,privacy_paragraphs), reverse=True)][:5]
		# Print the privacy-related paragraphs
		k=0
		string_list=[]
		for paragraph in top_paragraphs:
			k=k+1
			response = openai.Completion.create(
				model="text-davinci-003",
				prompt=f"Based on this privacy statement: {paragraph}, can you give me a privacy related question and 4 options with single correct answer and evidence saying it is correct from the given content? The question should be based on the given content and the format of the response should be like Question: Options: Answer: Evidence:.",
				temperature=0.7,
				max_tokens=256,
				top_p=1,
				frequency_penalty=0,
				presence_penalty=0
			)
			if response is not None:
				text=response["choices"][0]["text"]
				lines = text.split('\n\n')
				string_list.append(lines)  
		b=[]
		for i in string_list:
			for j in i:
				if len(j)==0:
					continue
				b.append(j)
		c=[]
		for i in b:
			k=str(i)
			sk=k.split("\n")
			c.append(sk)
		d=[]
		for i in c:
			for j in i:
				d.append(j)
		count=5
		temp=[]
		questions=[]
		answers=[]
		evidence=[]
		options=[]
		for i in d:
			if count<5:
				count=count+1
				temp.append(i)
			elif 'Question:' in i:
				questions.append(i)
			elif 'Options:' in i:
				count=0
			elif 'Answer' in  i:
				answers.append(i)
			elif 'Evidence' in i:
				evidence.append(i)
			if count==4:
				count=count+1
				options.append(temp)
				temp=[] 
		mykeys=[]
		for i in answers:
			temp=7
			count=0
			for j in i:
				s1=""+j
				if count<temp:
					count=count+1
					continue
				if s1.isalpha():
					j=j.lower()
					if j=='a':
						mykeys.append(0) 
					elif j=='b':
						mykeys.append(1)
					elif j=='c':
						mykeys.append(2)
					elif j=='d':
						mykeys.append(3) 
					break;  	
		print(questions)
		print(answers)
		print(check_list)
		print(mykeys)
	return render_template('details.html',
						check_list=check_list,
						questions=questions,
						options=options,
						correct_answers=mykeys,
						evidence=evidence)

if __name__ == '__main__':
    app.run()





