import pickle
import joblib

with open('tokenizer.pkl', 'rb') as f:
    obj = pickle.load(f)

joblib.dump(obj, 'tokenizer.joblib')