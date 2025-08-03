

# 🎬 Movie Recommendation System
A system features the lastest and most popular movies in real-time with a simple chatbot that could find similar movies based on your input!

![image](https://github.com/user-attachments/assets/67c9c0e1-33e9-4616-9571-04858a802eb4)
![image](https://github.com/user-attachments/assets/33da1f07-f090-4964-9680-4940293d0527)
![image](https://github.com/user-attachments/assets/076362c9-778b-4871-8920-a859df5ae5a1)
![image](https://github.com/user-attachments/assets/52cce7be-d120-415e-a3fb-6e18563755df)


## 🚀 Features

* 💬 Chat-like interface for movie recommendations
* 🔍 Search by movie name and get similar suggestions
* 🧠 Backend powered by TMDB’s recommendation engine
* ⚡ Fast, minimal UI with no reloads (AJAX-based)
* 🎨 Clean, responsive design using HTML/CSS/JavaScript


## 🛠️ Tech Stack

* **Frontend**: HTML, CSS, Vanilla JavaScript
* **Backend**: Python, Flask
* **API**: [TMDB (The Movie Database)](https://www.themoviedb.org/)

## 🧑‍💻 Installation

1. **Clone the repo:**

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Create a virtual environment & install dependencies:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

3. **Set your TMDB API key:**

   In `config.py` or `.env`:

   ```python
   api_key = "your_tmdb_api_key"
   ```

4. **Run the app:**

   ```bash
   py main.py
   ```

   Visit [http://127.0.0.1:5000/chatbot](http://127.0.0.1:5000/chatbot)

## 📁 Project Structure

```
├── static/
├── templates/
│   └── chat.html         # Main UI for the chatbot
├── app.py / main.py      # Flask application
├── recommender.py        # TMDB API integration logic
└── README.md
```

## ⚡ Future Improvements

* 🤖 Add GPT-style text generation for chatbot tone
* 🧠 Enhance with NLP to detect genres, actors, or moods
* 📱 Make fully mobile-friendly

## 🤝 Contributions

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

**Star ⭐ this project if you found it useful!**
Happy coding & movie hunting 🍿

---

Let me know if you'd like a version in Vietnamese or want to add a GIF demo badge!
