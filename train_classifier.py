import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

with open('./data.pickle', 'rb') as file:
    data_dict = pickle.load(file)

max_sequence_length = 84
data_padded = []

for seq in data_dict['data']:
    padded_seq = seq[:max_sequence_length] + [0] * max(0, max_sequence_length - len(seq))
    data_padded.append(padded_seq)

data = np.array(data_padded)

labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Use RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)

model.fit(x_train, y_train)
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly!'.format(score * 100))

# Save the trained model using pickle
with open('model.p', 'wb') as file:
    pickle.dump({'model': model}, file)
