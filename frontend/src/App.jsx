import { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import './App.css'

function App() {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const recognitionRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  function speakAnswer(answer) {
    const speech = new SpeechSynthesisUtterance(answer);
    speech.lang = 'en-US'; 
    speech.rate = 1; 
    window.speechSynthesis.speak(speech);
  }

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (messages.length === 0) {
      setMessages([
        {
          type: 'bot',
          content: '👋 Welcome! Ask me any legal question, and I will provide answers based on Indian law.',
        },
      ]);
    }
  }, []);

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputText.trim()) return;

    const userMessage = { type: 'user', content: inputText };
    const currentInputText = inputText;

    setInputText('');
    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const response = await axios.post('http://localhost:5000/ask', {
        question: currentInputText,
      });
      
      const { answer } = response.data;
      
      setMessages((prev) => [
        ...prev,
        { type: 'bot', content: `Answer: ${answer}` },
      ]);

      if (answer) {
        speakAnswer(answer);
      }
    } catch (error) {
      console.error('Error fetching answer:', error);
      setMessages((prev) => [
        ...prev,
        { type: 'bot', content: 'An error occurred while fetching the answer. Please try again.' },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0A0F1E] p-4 text-white">
      <div className="max-w-4xl mx-auto glass rounded-2xl shadow-2xl overflow-hidden transition-all duration-300 hover:shadow-blue-500/10">
        <div className="p-4 border-b border-blue-500/10 bg-slate-900/50">
          <h1 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-blue-600">
            AI Law Assistant
          </h1>
        </div>

        <div className={`overflow-y-auto p-6 space-y-4 transition-all duration-300 ${
          messages.length > 0 ? 'min-h-[200px] max-h-[600px]' : 'h-[100px]'
        }`}>
          {messages.map((message, index) => (
            <div
              key={index}
              className={`message-animation flex ${
                message.type === 'user' ? 'justify-end' : 'justify-start'
              }`}
            >
              <div
                className={`max-w-[80%] p-4 rounded-xl ${
                  message.type === 'user'
                    ? 'bg-blue-600/20 border border-blue-500/20 text-blue-100'
                    : 'bg-slate-800/50 border border-slate-700/50 text-slate-100'
                } shadow-lg backdrop-blur-sm`}
              >
                {message.content}
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex justify-center fade-in">
              <div className="bg-blue-500/10 text-blue-400 px-4 py-2 rounded-full border border-blue-500/20">
                Processing...
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="p-4 bg-slate-900/50 border-t border-blue-500/10">
          <form onSubmit={handleSendMessage} className="flex gap-2 items-center">
            <input
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="Ask your legal question here..."
              className="flex-1 p-3 rounded-lg bg-blue-500/10 border border-blue-500/20 text-white
                       placeholder-blue-300/50 focus:outline-none focus:ring-2 focus:ring-blue-500/20
                       transition-all duration-200"
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={isLoading}
              className="px-6 py-3 bg-blue-500/20 text-blue-300 rounded-lg border border-blue-500/20
                       hover:bg-blue-500/30 disabled:bg-slate-800/50 disabled:border-slate-700/50
                       disabled:text-slate-500 disabled:cursor-not-allowed
                       transition-all duration-200 font-medium"
            >
              Send
            </button>
          </form>
        </div>
      </div>
    </div>
  )
}

export default App
