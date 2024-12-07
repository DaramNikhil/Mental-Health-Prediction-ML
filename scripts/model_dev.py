import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

def model_dev(filtered_data):
    X = filtered_data.drop("treatment", axis=1)
    y = filtered_data["treatment"].apply(lambda x: 1 if x == "Yes" else 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train = X_train.values.astype(np.float32)
    X_test = X_test.values.astype(np.float32)
    y_train = np.array(y_train).astype(np.int64)
    y_test = np.array(y_test).astype(np.int64)

    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, "model/RandomForestClassifier.joblib")
    rf_pred = rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    print(f"Random Forest accuracy: {rf_accuracy}")

    pnn = GaussianNB()
    pnn.fit(X_train, y_train)
    pnn_pred = pnn.predict(X_test)
    pnn_accuracy = accuracy_score(y_test, pnn_pred)
    print(f"PNN accuracy: {pnn_accuracy}")

    input_size = X_train.shape[1]
    hidden_size = 16
    output_size = 2

    model = RNNClassifier(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train_tensor = torch.tensor(X_train).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train)
    X_test_tensor = torch.tensor(X_test).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test)

    epochs = 20
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    with torch.no_grad():
        rnn_pred = model(X_test_tensor)
        _, predicted = torch.max(rnn_pred, 1)
        rnn_accuracy = accuracy_score(y_test_tensor.numpy(), predicted.numpy())
    print(f"RNN accuracy: {rnn_accuracy}")

    return {
        "Random Forest": rf_accuracy,
        "PNN": pnn_accuracy,
        "RNN": rnn_accuracy
    }
