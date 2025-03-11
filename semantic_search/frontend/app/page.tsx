'use client';
import React, { useState } from 'react';
import axios from 'axios';
import './styles.css';

interface Message {
  sender: 'user' | 'bot';
  text: string;
  recommendations?: string[]; // Add recommendations to the message type
}

const ChatPage: React.FC = () => {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);

  const handleSend = async () => {
    if (!input.trim()) return;

    // Add user message to the conversation
    const userMessage: Message = { sender: 'user', text: input };
    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);

    // Use only the last two user messages as context
    const contextMessages = updatedMessages
      .filter((m) => m.sender === 'user')
      .slice(-2)
      .map((m) => m.text);

    setLoading(true);
    try {
      const response = await axios.post('http://127.0.0.1:5000/search', {
        query: input,
        context: contextMessages,
      });
      const botText = response.data.answers[0] || 'Sorry, no response.';
      const recommendations = response.data.recommendations || []; // Extract recommendations
      const botMessage: Message = {
        sender: 'bot',
        text: botText,
        recommendations: recommendations.length > 0 ? recommendations : undefined, // Only include if present
      };
      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error(error);
      const errorMsg: Message = { sender: 'bot', text: 'Error: Could not fetch response.' };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setLoading(false);
      setInput('');
    }
  };

  return (
    <div className="container">
      <h1>Mental Health Chat</h1>
      <div className="chat-box">
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`message ${msg.sender === 'user' ? 'user-message' : 'bot-message'}`}
          >
            <p>{msg.text}</p>
            {msg.recommendations && (
              <div className="recommendations">
                <strong>Recommendations:</strong>
                <ul>
                  {msg.recommendations.map((rec, i) => (
                    <li key={i}>{rec}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ))}
        {loading && <p>Loading...</p>}
      </div>
      <div className="input-box">
        <input
          type="text"
          placeholder="Type your message..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') handleSend();
          }}
        />
        <button className="button" onClick={handleSend}>
          Send
        </button>
      </div>
    </div>
  );
};

export default ChatPage;