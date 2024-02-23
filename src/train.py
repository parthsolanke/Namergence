import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model import RNN
from utils import loader

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

# Load the data
names, categories = loader.load_data()

# Hyper-parameters
learning_rate = 0.001
n_hidden = 128
n_iters = 100000
print_every = 5000
plot_every = 1000
current_loss = 0
all_losses = []
n_categories = len(categories)

# Initialize the model
rnn = RNN(n_categories, n_hidden, n_categories).to(device)
criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

# Category from output
def category_from_output(output):
    catagory_idx = torch.argmax(output).item()
    return categories[catagory_idx]


# Prediction function
def predict(name):
    with torch.no_grad():
        name_tensor = loader.line_to_tensor(name)
        hidden = rnn.init_hidden()
        for i in range(name_tensor.size()[0]):
            output, hidden = rnn(name_tensor[i], hidden)  
        label = category_from_output(output)
        confidence = torch.exp(output).max().item()
        return label, confidence


def train(name_tensor, label_tensor):
    hidden = rnn.init_hidden()
    for i in range(name_tensor.size()[0]):
        output, hidden = rnn(name_tensor[i], hidden)
    loss = criterion(output, label_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return output, loss.item()

# Train the model
for iter in range(1, n_iters + 1):
    category, name, label_tensor, name_tensor = loader.random_training_example(names, categories)
    output, loss = train(name_tensor, label_tensor)
    current_loss += loss

    if iter % print_every == 0:
        guess = category_from_output(output)
        correct = '✅' if guess == category else f'❌ ({category})'
        print(f'{iter} {iter/n_iters*100:.2f}% ({name}) {loss:.4f} {name} / {guess} {correct}')

    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0
        
        
# Plot the loss
plt.figure()
plt.plot(all_losses)
plt.show()

# Evaluation accuracy
with torch.no_grad():
    n_correct = 0
    n_total = 0
    for category in categories:
        for name in names[category]:
            label, confidence = predict(name)
            n_total += 1
            if label == category:
                n_correct += 1
    print(f'accuracy: {n_correct/n_total:.2f}')