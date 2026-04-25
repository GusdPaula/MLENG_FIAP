from joblib import load

model = load('.\\models\\best_model_with_metadata.pkl')

print(model['model_object'])
