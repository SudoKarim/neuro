import numpy as np

text_data = [
    ["I", "love", "to", "code"],
    ["We", "need", "to", "learn"],
    ["Time", "to", "go", "home"],
    ["Nice", "to", "meet", "you"]
]

def create_vocab(data):
    vocab = {}
    reverse_vocab = {}
    count = 0
    for sequence in data:
        for word in sequence:
            if word not in vocab:
                vocab[word] = count
                reverse_vocab[count] = word
                count += 1
    return vocab, reverse_vocab

def one_hot_encode(word_idx, vocab_size):
    encoding = np.zeros((vocab_size,))
    encoding[word_idx] = 1
    return encoding

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
        self.hidden_size = hidden_size
        
    def forward(self, inputs):
        h = np.zeros((self.hidden_size, 1))
        self.xs, self.hs, self.ys, self.ps = {}, {}, {}, {}
        self.hs[-1] = np.copy(h)
        for t in range(len(inputs)):
            self.xs[t] = inputs[t].reshape(-1, 1)
            self.hs[t] = np.tanh(np.dot(self.Wxh, self.xs[t]) + 
                                np.dot(self.Whh, self.hs[t-1]) + self.bh)
            self.ys[t] = np.dot(self.Why, self.hs[t]) + self.by
            self.ps[t] = np.exp(self.ys[t]) / np.sum(np.exp(self.ys[t]))
            
        return self.ps[len(inputs)-1]
    
    def backward(self, inputs, target, learning_rate=0.01):
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        dhnext = np.zeros_like(self.hs[0])
        dy = np.copy(self.ps[len(inputs)-1])
        dy[target] -= 1
        dWhy += np.dot(dy, self.hs[len(inputs)-1].T)
        dby += dy
        for t in reversed(range(len(inputs))):
            dh = np.dot(self.Why.T, dy) + dhnext
            dhraw = (1 - self.hs[t] * self.hs[t]) * dh
            dbh += dhraw
            dWxh += np.dot(dhraw, self.xs[t].T)
            dWhh += np.dot(dhraw, self.hs[t-1].T)
            dhnext = np.dot(self.Whh.T, dhraw)
        self.Wxh -= learning_rate * dWxh
        self.Whh -= learning_rate * dWhh
        self.Why -= learning_rate * dWhy
        self.bh -= learning_rate * dbh
        self.by -= learning_rate * dby
def train_rnn():
    vocab, reverse_vocab = create_vocab(text_data)
    vocab_size = len(vocab)
    rnn = SimpleRNN(input_size=vocab_size, hidden_size=50, output_size=vocab_size)
    epochs = 1000
    for epoch in range(epochs):
        total_loss = 0

        for sequence in text_data:
            inputs = [one_hot_encode(vocab[word], vocab_size) for word in sequence[:3]]
            target = vocab[sequence[3]]
            prob = rnn.forward(inputs)
            loss = -np.log(prob[target])
            total_loss += loss
            rnn.backward(inputs, target)
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss/len(text_data)}')
    
    return rnn, vocab, reverse_vocab

print("Training the RNN...")
rnn, vocab, reverse_vocab = train_rnn()

def predict_fourth_word(rnn, first_three_words, vocab, reverse_vocab):
    inputs = [one_hot_encode(vocab[word], len(vocab)) for word in first_three_words]
    prob = rnn.forward(inputs)
    predicted_idx = np.argmax(prob)
    return reverse_vocab[predicted_idx]

test_sequence = ["I", "love", "to"]
predicted = predict_fourth_word(rnn, test_sequence, vocab, reverse_vocab)
print(f"\nTest prediction:")
print(f"Input: {' '.join(test_sequence)}")
print(f"Predicted fourth word: {predicted}")
