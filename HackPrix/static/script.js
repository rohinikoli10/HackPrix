document.addEventListener('DOMContentLoaded', function() {
    const userInfo = JSON.parse(localStorage.getItem('userInfo'));
    if (userInfo) {
        document.getElementById('user-name').textContent = userInfo.name;
        document.getElementById('user-bp').textContent = `BP: ${userInfo.bp}`;
        document.getElementById('user-heart-rate').textContent = `Heart Rate: ${userInfo.heartRate} bpm`;
        document.getElementById('user-sugar-level').textContent = `Sugar Level: ${userInfo.sugarLevel} mg/dL`;
    }
});

document.getElementById('send-btn').addEventListener('click', handleUserInput);

document.getElementById('chat-input').addEventListener('keydown', function(event) {
    if (event.keyCode === 13) {
        event.preventDefault();
        handleUserInput();
    }
});

function handleUserInput() {
    const userInput = document.getElementById('chat-input').value;
    if (userInput.trim() !== '') {
        addMessage(userInput, 'outgoing');
        document.getElementById('chat-input').value = '';

        // Check if user input is 'hi', then reply with assistance message
        if (userInput.toLowerCase() === 'hi') {
            setTimeout(() => {
                addMessage('I am here to assist you with your health queries.', 'incoming');
            }, 1000);
            return;
        }

        // Otherwise, send symptoms to server for prediction
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ symptoms: userInput.split(',').map(symptom => symptom.trim()) })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            data.predictions.forEach(prediction => {
                addMessage(`Predicted Disease: ${prediction.disease}`, 'incoming');
                addMessage(`Description: ${prediction.description}`, 'incoming');
                addMessage(`Precautions: ${prediction.precautions.join(', ')}`, 'incoming');
            });
        })
        .catch(error => {
            console.error('Error:', error);
            addMessage('Error: Unable to predict disease', 'incoming');
        });
    }
}

function addMessage(text, type) {
    const chatContainer = document.querySelector('.chat-container');
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat ${type}`;
    const messageContent = document.createElement('div');
    messageContent.className = 'chat-details';
    const messageText = document.createElement('p');
    messageText.textContent = text;
    messageContent.appendChild(messageText);
    messageDiv.appendChild(messageContent);
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function toggleProfileDropdown() {
    const dropdown = document.getElementById('profile-dropdown');
    dropdown.classList.toggle('show');
}

function toggleDarkMode() {
    document.body.classList.toggle('light-mode');
    const themeBtn = document.getElementById('theme-btn');
    if (document.body.classList.contains('light-mode')) {
        themeBtn.textContent = 'dark_mode';
    } else {
        themeBtn.textContent = 'light_mode';
    }
}

function deleteChats() {
    const chatContainer = document.querySelector('.chat-container');
    chatContainer.innerHTML = '';
}
