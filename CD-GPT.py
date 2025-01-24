import nltk
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import json
import pickle
import random

nltk.download('punkt')
nltk.download('wordnet')

nltk.download('punkt_tab') 

from nltk.stem import WordNetLemmatizer

class CDGPT:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.load_intents()
        self.preprocess_training_data()
        self.build_model()
    def load_intents(self):
        """
        i will design for college recommendation system for students , 
        Load predefined intents for educational guidance
        Covers undergraduate, postgraduate, courses, fees, jobs, colleges
        """
        self.intents = {
            "greeting": [
                "hi", "hello", "hey", "greetings",
                "Ram Ram", "Sir ji", 
                "good morning", "good afternoon"
            ],
            "undergraduate_courses": [
                "what undergraduate courses are available?", 
                "list of ug courses", 
                "After 12 th , what we do in graduates ?",
                "undergraduate degree options"
            ],
            "postgraduate_courses": [
                "postgraduate courses", 
                "pg courses", 
                "masters degree options"
            ],
            "fee_structure": [
                "course fees", 
                "fee structure", 
                "how much does it cost?"
            ],
            "job_prospects": [
                "career opportunities", 
                "job market", 
                "future jobs", 
                "demanding careers"
            ],
            "college_info": [
                "best colleges", 
                "top universities", 
                "college recommendations"
            ],
            "course_guidance": [
                "help me choose a course", 
                "course selection advice", 
                "which course is best for me?"
            ],
            "goodbye": [
                "bye", "goodbye", "exit", "close", "quit"
            ]
        }
        
        # Predefined responses accroding to questions
        self.responses = {
            "greeting": [
                "Welcome to CD-GPT! I'm here to guide you through educational opportunities.",
                "Hello! Ready to explore your academic future?",
                "Hi there! How can I help you with your educational journey today?"
            ],
            "undergraduate_courses": [
                "Popular undergraduate courses include: Computer Science, Engineering, Business Administration, Psychology, and Economics.",
                "Some top undergraduate degrees are: BTech, BA, BSc, BBA, and Bachelor of Commerce.",
                "from CollegeDekho , we provide best technical Courses and skills which are used in coporate sector such as- B.tech(AI & DS), B.tech(CS), BCA(CS/DS/FSD), BBA(DM) ",
                "Undergraduate options vary by field: STEM, Humanities, Business, and Social Sciences offer diverse paths."
            ],
            "postgraduate_courses": [
                "Postgraduate options include MBA, MTech, MS, MA in various specializations.",
                "Popular PG courses: Data Science, Artificial Intelligence, Digital Marketing, and International Business.",
                "Masters degrees offer deep specialization in fields like Engineering, Management, and Research."
            ],
            "fee_structure": [
                "Fees vary by institution and course. Expect ranges: UG (₹50,000 - ₹5,00,000), PG (₹1,00,000 - ₹10,00,000).",
                "Government colleges are more affordable. Private institutions have higher fees but offer specialized programs.",
                "in Sigma University fee structure is: - 1. BCA(CS/DS) = ₹90,000, 2. B.tech(AI & DS) or B.tech(CS)= ₹1,50,000",
                "Consider scholarships, education loans, and financial aid to manage educational expenses."
            ],
            "job_prospects": [
                "High-demand fields: Technology, Data Science, Digital Marketing, Cybersecurity, and Healthcare.",
                "IT, AI, Machine Learning, and Cloud Computing offer excellent career opportunities.",
                "Emerging careers: Blockchain Developer, UX Designer, Digital Transformation Specialist."
            ],
            "college_info": [
                "Top institutions: IITs, NITs, IIMs, Delhi University, Bangalore University.",
                "We Provides top institutions: Sigma University Vadodara, Doon University, Rai University, etc",
                "Consider factors: Academic reputation, placement record, faculty, infrastructure, and research opportunities.",
                "Research colleges through official websites, national ranking platforms, and student forums."
            ],
            "course_guidance": [
                "Consider your interests, strengths, and career goals. Assess your academic performance and passion.",
                "Take aptitude tests, consult career counselors, and explore internship opportunities.",
                "Align your course with future job market trends and personal aspirations."
            ],
            "goodbye": [
                "Thank you for using CD-GPT. Wish you the best in your educational journey!",
                "Goodbye! Remember, your future is full of possibilities.",
                "Stay curious and keep learning. Bye for now!"
            ]
        }
    
    def preprocess_text(self, text):
        tokens = nltk.word_tokenize(text.lower())
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return " ".join(lemmatized_tokens)
  
    def preprocess_training_data(self):
        training_samples = []
        training_labels = []
        for intent, patterns in self.intents.items():
            for pattern in patterns:
                training_samples.append(self.preprocess_text(pattern))
                training_labels.append(intent)
        self.tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(training_samples)
        sequences = self.tokenizer.texts_to_sequences(training_samples)
        padded_sequences = pad_sequences(sequences, maxlen=20, padding='post', truncating='post')
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(training_labels)
        self.X_train = np.array(padded_sequences)
        self.y_train = tf.keras.utils.to_categorical(encoded_labels)
    
    def build_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(1000, 16, input_length=20),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(len(self.intents), activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam', 
            loss='categorical_crossentropy', 
            metrics=['accuracy']
        )
        
        # Train the model
        self.model.fit(
            self.X_train, self.y_train, 
            epochs=50, 
            validation_split=0.2
        )
    
    def predict_intent(self, user_input):
        processed_input = self.preprocess_text(user_input)
        sequence = self.tokenizer.texts_to_sequences([processed_input])
        padded_sequence = pad_sequences(sequence, maxlen=20, padding='post', truncating='post')
        prediction = self.model.predict(padded_sequence)
        predicted_intent = self.label_encoder.inverse_transform([np.argmax(prediction)])
        return predicted_intent[0]

    def get_response(self, intent):
        return random.choice(self.responses.get(intent, ["I'm not sure how to respond to that."]))
    def chat(self):
        print("CD-GPT: Welcome! I'm your educational guidance assistant. Type 'bye' to exit.")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['bye', 'goodbye', 'exit', 'quit']:
                print("CD-GPT: " + random.choice(self.responses['goodbye']))
                break
            intent = self.predict_intent(user_input)
            response = self.get_response(intent)
            print(f"CD-GPT: {response}")
if __name__ == "__main__":
    chatbot = CDGPT()
    chatbot.chat()