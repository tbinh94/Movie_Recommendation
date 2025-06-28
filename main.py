import os
import sqlite3
import requests
import pandas as pd
from typing import List, Dict
from flask import Flask, request, render_template, redirect, url_for, session, flash, abort
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import jsonify

# ----------------------------------------
# 1. Load TMDB API Key từ api.env
# ----------------------------------------
load_dotenv(dotenv_path="api.env")
api_key = os.getenv("TMDB_API_KEY")
print(f"Loaded TMDB API Key: {'Yes' if api_key else 'No - Check api.env'}")

# ----------------------------------------
# 2. Khởi tạo Flask & cấu hình session secret key
# ----------------------------------------
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-secret-key")  # Đổi "your-secret-key" thành một chuỗi bí mật thật
# Đảm bảo bạn có FLASK_SECRET_KEY trong api.env để bảo mật hơn

# ----------------------------------------
# 3. Định nghĩa class MovieRecommendationSystem
# ----------------------------------------
class MovieRecommendationSystem:
    def __init__(self):
        self.movies_df = None
        self.ratings_df = None
        self.conn = None
        self.setup_database()
        self.load_sample_data()

    def setup_database(self):
        """Tạo database SQLite và các bảng nếu chưa tồn tại."""
        try:
            self.conn = sqlite3.connect('database\movie_recommendations.db', check_same_thread=False)
            cursor = self.conn.cursor()
            
            # Bảng movies
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS movies (
                    id INTEGER PRIMARY KEY,
                    title TEXT NOT NULL,
                    genre TEXT,
                    year INTEGER,
                    description TEXT,
                    rating REAL DEFAULT 0.0
                )
            ''')
            
            # Bảng user_ratings - Fix UNIQUE constraint
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_ratings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    movie_id INTEGER,
                    rating REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (movie_id) REFERENCES movies (id)
                )
            ''')
            
            # Tạo unique index riêng nếu chưa có
            cursor.execute('''
                CREATE UNIQUE INDEX IF NOT EXISTS idx_user_movie 
                ON user_ratings (user_id, movie_id)
            ''')
            
            # Bảng users
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                is_admin BOOLEAN DEFAULT 0,
                avatar_url TEXT DEFAULT 'https://example.com/default-avatar.png'
                )
            ''')
            
            self.conn.commit()
            print("Database setup completed successfully")
            
        except sqlite3.Error as e:
            print(f"Database setup error: {e}")
            raise

    def find_movie_by_title(self, title: str) -> int:
        df = self.movies_df
        match = df[df['title'].str.contains(title, case=False, na=False)]
        if not match.empty:
            return int(match.iloc[0]['id'])
        tmdb = self.search_movie_tmdb(title)
        if tmdb:
            return tmdb[0]['id']
        raise ValueError(f"Movie '{title}' not found.")

    def get_tmdb_recommendations(self, tmdb_id: int, max_pages: int = 1) -> List[Dict]:
            """
            Gọi TMDB API /movie/{tmdb_id}/recommendations
            """
            if not api_key:
                return []
            base_url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/recommendations"
            all_results: List[Dict] = []
            for page in range(1, max_pages + 1):
                params = {
                    "api_key": api_key,
                    "language": "en-US",
                    "page": page
                }
                resp = requests.get(base_url, params=params)
                if resp.status_code != 200:
                    print(f"[WARN] TMDB recommendations status {resp.status_code}")
                    break
                data = resp.json()
                for m in data.get("results", []):
                    poster = m.get("poster_path")
                    poster_url = f"https://image.tmdb.org/t/p/w342{poster}" if poster else None
                    all_results.append({
                        "id": m["id"],
                        "title": m["title"],
                        "year": m.get("release_date", "")[:4] or "N/A",
                        "genre": "",
                        "rating": m.get("vote_average", 0),
                        "description": m.get("overview", ""),
                        "poster_url": poster_url,
                        "source": "TMDB"
                    })
                if page >= data.get("total_pages", 1):
                    break
            return all_results
    
    def get_tmdb_popular(self, max_pages: int = 1) -> List[Dict]:
        """
        Gọi TMDB API /movie/popular để lấy danh sách phim phổ biến.
        """
        if not api_key:
            return []
        all_results = []
        base_url = "https://api.themoviedb.org/3/movie/popular"
        for page in range(1, max_pages+1):
            params = {"api_key": api_key, "language": "en-US", "page": page}
            resp = requests.get(base_url, params=params)
            if resp.status_code != 200:
                break
            data = resp.json()
            for m in data.get("results", []):
                poster = m.get("poster_path")
                poster_url = f"https://image.tmdb.org/t/p/w342{poster}" if poster else None
                all_results.append({
                    "id": m["id"],
                    "title": m["title"],
                    "year": m.get("release_date","")[:4] or "N/A",
                    "rating": m.get("vote_average", 0),
                    "description": m.get("overview",""),
                    "poster_url": poster_url,
                    "source": "TMDB"
                })
            if page >= data.get("total_pages",1):
                break
        return all_results        
    
    def get_tmdb_genres(self) -> List[Dict]:
        """
        Lấy danh sách genre từ TMDB
        """
        if not api_key:
            return []
        url = "https://api.themoviedb.org/3/genre/movie/list"
        resp = requests.get(url, params={"api_key": api_key, "language": "en-US"})
        data = resp.json() if resp.status_code == 200 else {}
        return data.get("genres", [])  # mỗi genre: {"id": 28, "name": "Action"}

    def get_tmdb_movies_by_genre(self, genre_id: int, max_pages: int = 1) -> List[Dict]:
        """
        Dùng TMDB Discover để lấy phim theo thể loại
        """
        if not api_key:
            return []
        url = "https://api.themoviedb.org/3/discover/movie"
        results = []
        for page in range(1, max_pages+1):
            params = {
                "api_key": api_key,
                "language": "en-US",
                "with_genres": genre_id,
                "sort_by": "popularity.desc",
                "page": page
            }
            resp = requests.get(url, params=params)
            if resp.status_code != 200:
                break
            data = resp.json()
            for m in data.get("results", []):
                poster = m.get("poster_path")
                results.append({
                    "id": m["id"],
                    "title": m["title"],
                    "year": m.get("release_date","")[:4] or "N/A",
                    "rating": m.get("vote_average", 0),
                    "description": m.get("overview",""),
                    "poster_url": f"https://image.tmdb.org/t/p/w342{poster}" if poster else None,
                    "source": "TMDB"
                })
            if page >= data.get("total_pages",1):
                break
        return results
    
    def load_sample_data(self):
        """Chèn sample data vào bảng movies và user_ratings, và nạp vào pandas DataFrame."""
        sample_movies = [
            (1, "The Shawshank Redemption", "Drama", 1994,
             "Two imprisoned men bond over years, finding solace and redemption", 9.3),
            (2, "The Godfather", "Crime,Drama", 1972,
             "The aging patriarch of an organized crime dynasty transfers control", 9.2),
            (3, "The Dark Knight", "Action,Crime,Drama", 2008,
             "Batman faces the Joker, a criminal mastermind", 9.0),
            (4, "Pulp Fiction", "Crime,Drama", 1994,
             "The lives of two mob hitmen, a boxer and others intertwine", 8.9),
            (5, "Forrest Gump", "Drama,Romance", 1994,
             "The presidencies of Kennedy and Johnson through Vietnam War", 8.8),
            (6, "Inception", "Action,Sci-Fi,Thriller", 2010,
             "A thief who steals corporate secrets through dream-sharing", 8.7),
            (7, "The Matrix", "Action,Sci-Fi", 1999,
             "A computer hacker learns about the true nature of reality", 8.7),
            (8, "Goodfellas", "Biography,Crime,Drama", 1990,
             "The story of Henry Hill and his life in the mob", 8.7),
            (9, "Interstellar", "Adventure,Drama,Sci-Fi", 2014,
             "A team of explorers travel through a wormhole in space", 8.6),
            (10, "The Lion King", "Animation,Adventure,Drama", 1994,
             "A young lion prince flees his kingdom", 8.5)
        ]
        cursor = self.conn.cursor()
        cursor.executemany('INSERT OR REPLACE INTO movies VALUES (?,?,?,?,?,?)', sample_movies)

        sample_ratings = [
            (1, 1, 5.0), (1, 2, 4.5), (1, 3, 4.8), (1, 6, 4.9), (1, 7, 4.7),
            (2, 1, 4.8), (2, 4, 5.0), (2, 5, 3.5), (2, 8, 4.6), (2, 9, 4.3),
            (3, 2, 4.9), (3, 3, 5.0), (3, 6, 4.8), (3, 7, 4.9), (3, 10, 3.8),
            (4, 1, 4.2), (4, 5, 4.8), (4, 9, 5.0), (4, 10, 4.3), (4, 4, 3.9),
            (5, 3, 4.7), (5, 6, 4.5), (5, 7, 4.8), (5, 8, 4.1), (5, 2, 4.6)
        ]
        cursor.execute('DELETE FROM user_ratings')
        cursor.executemany('INSERT INTO user_ratings (user_id, movie_id, rating) VALUES (?,?,?)', sample_ratings)
        self.conn.commit()

        # Đọc vào DataFrame để xử lý nội bộ
        self.movies_df = pd.read_sql_query("SELECT * FROM movies", self.conn)
        self.ratings_df = pd.read_sql_query("SELECT * FROM user_ratings", self.conn)

    def search_movie_tmdb(self, query: str, max_pages: int = 1) -> List[Dict]:
        """
        Gọi TMDB API để tìm movie theo query.
        Trả về list[dict] với keys: id, title, year, genre, rating, description, poster_url, source="TMDB".
        """
        all_results: List[Dict] = []
        if not api_key:
            return []

        base_url = "https://api.themoviedb.org/3/search/movie"
        for page in range(1, max_pages + 1):
            params = {
                "api_key": api_key,
                "query": query,
                "language": "en-US",
                "page": page
            }
            try:
                resp = requests.get(base_url, params=params)
                if resp.status_code != 200:
                    break
                data = resp.json()
                results = data.get("results", [])
                if not results:
                    break

                for m in results:
                    poster_path = m.get("poster_path")
                    if poster_path:
                        poster_url = f"https://image.tmdb.org/t/p/w342{poster_path}"
                    else:
                        poster_url = None

                    all_results.append({
                        "id": m.get("id"),
                        "title": m.get("title"),
                        "year": m.get("release_date", "")[:4] if m.get("release_date") else "N/A",
                        "genre": "",
                        "rating": m.get("vote_average", 0),
                        "description": m.get("overview", ""),
                        "poster_url": poster_url,
                        "source": "TMDB"
                    })

                total_pages = data.get("total_pages", 1)
                if page >= total_pages:
                    break

            except requests.exceptions.RequestException:
                break

        return all_results

    def search_movies(self, keyword: str, use_tmdb_checkbox: bool = True) -> List[Dict]:
        """
        Hàm tìm kiếm: nếu checkbox 'Use TMDB' được tick và có api_key, gọi TMDB,
        ngược lại fallback sang tìm kiếm local DB.
        """
        results: List[Dict] = []

        if use_tmdb_checkbox and api_key:
            tmdb_raw = self.search_movie_tmdb(keyword, max_pages=1)
            if tmdb_raw:
                return tmdb_raw

        # Fallback sang local DB
        cursor = self.conn.cursor()
        search_term = f"%{keyword}%"
        cursor.execute("""
            SELECT id, title, genre, year, description, rating
            FROM movies
            WHERE title LIKE ? OR description LIKE ?
        """, (search_term, search_term))
        local_rows = cursor.fetchall()

        for row in local_rows:
            results.append({
                "id": row[0],
                "title": row[1],
                "genre": row[2],
                "year": row[3],
                "description": row[4],
                "rating": row[5],
                "poster_url": None,
                "source": "Local DB"
            })
        return results

        

    def content_based_recommendation(self, movie_id: int, num_recommendations: int = 5) -> List[Dict]:
        """Sinh khuyến nghị content-based dựa trên cosine similarity genre+description."""
        if self.movies_df is None or self.movies_df.empty:
            return []
        if movie_id not in self.movies_df['id'].values:
            return []

        self.movies_df['content'] = self.movies_df['genre'].fillna('') + ' ' + self.movies_df['description'].fillna('')
        tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
        content_matrix = tfidf.fit_transform(self.movies_df['content'])
        sim_matrix = cosine_similarity(content_matrix)

        try:
            idx = self.movies_df[self.movies_df['id'] == movie_id].index[0]
        except IndexError:
            return []

        sim_scores = list(enumerate(sim_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        recommendations = []
        for i in range(1, min(len(sim_scores), num_recommendations + 1)):
            idx_movie = sim_scores[i][0]
            if idx_movie < len(self.movies_df):
                movie = self.movies_df.iloc[idx_movie]
                recommendations.append({
                    'id': movie['id'],
                    'title': movie['title'],
                    'genre': movie['genre'],
                    'year': movie['year'],
                    'rating': movie['rating'],
                    'similarity_score': sim_scores[i][1]
                })
            if len(recommendations) >= num_recommendations:
                break
        return recommendations

    def collaborative_filtering(self, user_id: int, num_recommendations: int = 5) -> List[Dict]:
        """Sinh khuyến nghị collaborative filtering dựa trên cosine similarity giữa user-user."""
        if self.ratings_df is None or self.ratings_df.empty:
            return self.get_popular_movies(num_recommendations)

        user_movie_matrix = self.ratings_df.groupby(['user_id', 'movie_id'])['rating'].mean().unstack(fill_value=0)
        if user_id not in user_movie_matrix.index:
            return self.get_popular_movies(num_recommendations)

        user_sim = cosine_similarity(user_movie_matrix)
        user_sim_df = pd.DataFrame(user_sim, index=user_movie_matrix.index, columns=user_movie_matrix.index)
        similar_users = user_sim_df[user_id].sort_values(ascending=False).drop(user_id, errors='ignore').head(5)

        recommendations = []
        user_rated = set(self.ratings_df[self.ratings_df['user_id'] == user_id]['movie_id'].values)

        for other_user, sim_score in similar_users.items():
            if sim_score <= 0:
                continue
            rows = self.ratings_df[
                (self.ratings_df['user_id'] == other_user) &
                (self.ratings_df['rating'] >= 4.0) &
                (~self.ratings_df['movie_id'].isin(user_rated))
            ]
            for _, row in rows.iterrows():
                movie_info = self.movies_df[self.movies_df['id'] == row['movie_id']]
                if not movie_info.empty:
                    m = movie_info.iloc[0]
                    recommendations.append({
                        'id': m['id'],
                        'title': m['title'],
                        'genre': m['genre'],
                        'year': m['year'],
                        'rating': m['rating'],
                        'predicted_rating': row['rating'] * sim_score
                    })

        if recommendations:
            rec_df = pd.DataFrame(recommendations)
            rec_df = rec_df.sort_values('predicted_rating', ascending=False)
            rec_df = rec_df.drop_duplicates(subset=['id'], keep='first')
            return rec_df.head(num_recommendations).to_dict('records')

        return self.get_popular_movies(num_recommendations)

    def hybrid_recommendation(self, user_id: int, movie_id: int = None, num_recommendations: int = 5) -> Dict:
        """
        Kết hợp cả content-based và collaborative để tạo khuyến nghị.
        Trả về dict: {'content_based': [...], 'collaborative': [...], 'hybrid': [...]}
        """
        recs = {'content_based': [], 'collaborative': [], 'hybrid': []}

        if movie_id:
            if self.movies_df is not None and movie_id in self.movies_df['id'].values:
                recs['content_based'] = self.content_based_recommendation(movie_id, num_recommendations)
            else:
                print(f"Movie ID {movie_id} not found for content-based.")

        recs['collaborative'] = self.collaborative_filtering(user_id, num_recommendations)

        hybrid_dict = {}
        # Add content-based với weight 0.4
        for movie in recs['content_based']:
            hybrid_dict[movie['id']] = {
                **movie,
                'hybrid_score': movie.get('similarity_score', 0) * 0.4
            }
        # Add collaborative với weight 0.6
        for movie in recs['collaborative']:
            mid = movie['id']
            if mid in hybrid_dict:
                hybrid_dict[mid]['hybrid_score'] += movie.get('predicted_rating', 0) * 0.6
                for k, v in movie.items():
                    if k not in hybrid_dict[mid] or hybrid_dict[mid][k] is None:
                        hybrid_dict[mid][k] = v
            else:
                hybrid_dict[mid] = {
                    **movie,
                    'hybrid_score': movie.get('predicted_rating', 0) * 0.6
                }

        recs['hybrid'] = sorted(hybrid_dict.values(), key=lambda x: x.get('hybrid_score', 0), reverse=True)[:num_recommendations]
        return recs

    def add_user_rating(self, user_id: int, movie_id: int, rating: float):
        """Thêm hoặc cập nhật rating của user cho movie (upsert)."""
        cursor = self.conn.cursor()
        
        # Kiểm tra movie có tồn tại không
        cursor.execute("SELECT id FROM movies WHERE id = ?", (movie_id,))
        if cursor.fetchone() is None:
            raise ValueError(f"Movie with ID {movie_id} does not exist.")

        # Sử dụng INSERT OR REPLACE thay vì ON CONFLICT (SQLite không hỗ trợ ON CONFLICT với composite keys cũ)
        cursor.execute("""
            INSERT OR REPLACE INTO user_ratings (user_id, movie_id, rating, timestamp)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        """, (user_id, movie_id, rating))
        
        self.conn.commit()
        
        # Reload ratings dataframe
        self.ratings_df = pd.read_sql_query("SELECT * FROM user_ratings", self.conn)
        print(f"Rating {rating} for movie {movie_id} by user {user_id} added/updated.")

# ----------------------------------------
# 4. Khởi tạo recommender
# ----------------------------------------
recommender = MovieRecommendationSystem()

# ----------------------------------------
# 5. Route phía user (đăng ký, đăng nhập, đăng xuất)
# ----------------------------------------

@app.route('/register', methods=['GET', 'POST'])
def register():
    """
    Trang đăng ký: username, email, password, confirm password.
    - Kiểm tra username/email không trùng lặp.
    - Hash mật khẩu trước khi lưu vào DB.
    """
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm', '')

        # 1. Validation cơ bản
        if not username or not email or not password or not confirm:
            flash("Vui lòng điền đầy đủ các trường.", "danger")
            return render_template('register.html')
            
        # Validation độ dài
        if len(username) < 3:
            flash("Username phải có ít nhất 3 ký tự.", "danger")
            return render_template('register.html')
            
        if len(password) < 6:
            flash("Mật khẩu phải có ít nhất 6 ký tự.", "danger")
            return render_template('register.html')
            
        if password != confirm:
            flash("Mật khẩu và xác nhận mật khẩu không khớp.", "danger")
            return render_template('register.html')

        # 2. Kiểm tra email format cơ bản
        if '@' not in email or '.' not in email.split('@')[-1]:
            flash("Email không hợp lệ.", "danger")
            return render_template('register.html')

        # 3. Kiểm tra username và email đã tồn tại chưa
        cursor = recommender.conn.cursor()
        
        # Kiểm tra username
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        if cursor.fetchone():
            flash("Username đã được sử dụng.", "danger")
            return render_template('register.html')
            
        # Kiểm tra email
        cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
        if cursor.fetchone():
            flash("Email đã được sử dụng.", "danger")
            return render_template('register.html')

        # 4. Hash password và lưu user
        hashed = generate_password_hash(password)

        try:
            cursor.execute("""
                INSERT INTO users (username, email, password)
                VALUES (?, ?, ?)
            """, (username, email, hashed))
            recommender.conn.commit()
            flash("Đăng ký thành công! Vui lòng đăng nhập.", "success")
            return redirect(url_for('login'))
            
        except sqlite3.Error as e:
            print(f"Database error during registration: {e}")
            flash("Đã xảy ra lỗi khi đăng ký. Vui lòng thử lại.", "danger")
            return render_template('register.html')

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """
    Trang đăng nhập:
    - Kiểm tra username tồn tại chưa
    - Dùng check_password_hash để so sánh mật khẩu
    - Nếu đúng, lưu session['user_id'] và session['username']
    """
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        if not username or not password:
            flash("Vui lòng điền username và mật khẩu.", "danger")
            return render_template('login.html')

        try:
            cursor = recommender.conn.cursor()
            cursor.execute("SELECT id, password, avatar_url FROM users WHERE username = ?", (username,))
            row = cursor.fetchone()
            
            if row:
                user_id, hashed, avatar_url = row
                if check_password_hash(hashed, password):
                    # Đăng nhập thành công
                    session['user_id'] = user_id
                    session['username'] = username
                    session['avatar_url'] = avatar_url or 'https://example.com/default-avatar.png'
                    flash(f"Chào mừng {username}!", "success")
                    return redirect(url_for('index'))
                else:
                    flash("Sai mật khẩu.", "danger")
            else:
                flash("Username không tồn tại.", "danger")
                
        except sqlite3.Error as e:
            print(f"Database error during login: {e}")
            flash("Đã xảy ra lỗi khi đăng nhập. Vui lòng thử lại.", "danger")
        
    return render_template('login.html')


@app.route('/logout')
def logout():
    """Đăng xuất: xóa session và chuyển về trang index."""
    session.clear()
    flash("Bạn đã đăng xuất.", "info")
    return redirect(url_for('index'))

# ----------------------------------------
# 6. Các route chính (home, search, recommend, add_rating)
# ----------------------------------------

@app.route('/')
def index():
    """
    Trang chủ: hiển thị banner và featured movies (lấy 5 phim rating cao nhất).
    Với mỗi phim, lấy poster bằng TMDB (dùng search bằng title, page=1, kết quả đầu tiên).
    """
    featured_movies = recommender.get_tmdb_popular(max_pages=1)[:20]

    # Tạo cache tạm thời để tránh gọi nhiều lần cùng một title
    poster_cache = {}
    for movie in featured_movies:
        title = movie.get("title", "").strip()
        if title in poster_cache:
            movie["poster_url"] = poster_cache[title]
        else:
            tmdb_res = recommender.search_movie_tmdb(title, max_pages=1)
            if tmdb_res and tmdb_res[0].get("poster_url"):
                poster_url = tmdb_res[0]["poster_url"]
            else:
                poster_url = None
            poster_cache[title] = poster_url
            movie["poster_url"] = poster_url

    return render_template('index.html', featured_movies=featured_movies)


@app.route('/search', methods=['GET'])
def search():
    """
    Trang tìm kiếm phim:
    - Nếu query không rỗng, gọi recommender.search_movies(query, use_tmdb_checkbox)
    - Kết quả trả về đã bao gồm poster_url, rating, description, source.
    """
    query = request.args.get('q', '').strip()
    results_list = None

    if query:
        results_list = recommender.search_movie_tmdb(query, max_pages=1)

    return render_template('search.html',
                           results=results_list,
                           query=query)


"""
@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    # 1. Bắt buộc login
    if 'user_id' not in session:
        flash("Vui lòng đăng nhập để lấy gợi ý.", "danger")
        return redirect(url_for('login'))
    user_id = session['user_id']

    recs = {}
    error = None

    # 2. Lấy tmdb_id từ GET (click poster), base_movie_id từ POST (form)
    tmdb_id_str     = request.args.get('tmdb_id')
    base_movie_id   = request.form.get('base_movie_id')
    num_str         = request.form.get('num') or request.args.get('num') or '10'
    try:
        num = int(num_str)
    except ValueError:
        num = 10

    # 3. Xử lý GET với tmdb_id → TMDB Online Recommendations
    if request.method == 'GET' and tmdb_id_str:
        try:
            tmdb_id = int(tmdb_id_str)
        except ValueError:
            tmdb_id = None
        if tmdb_id:
            try:
                # Thử gọi TMDB recommendations endpoint
                print(f"[DEBUG] Calling TMDB /movie/{tmdb_id}/recommendations")
                recs['tmdb_online'] = recommender.get_tmdb_recommendations(tmdb_id, max_pages=1)[:num]
                print(f"[DEBUG] TMDB returned {len(recs['tmdb_online'])} items")
            except Exception as e:
                print(f"[ERROR] TMDB recommend exception: {e}")
                error = "Không thể lấy gợi ý online từ TMDB. Vui lòng thử lại sau."
        else:
            error = "TMDB ID không hợp lệ."
    # 4. Xử lý POST → hybrid/local như cũ
    elif request.method == 'POST':
        movie_id = None
        if base_movie_id:
            try:
                movie_id = int(base_movie_id)
            except ValueError:
                error = "Phải chọn một phim hợp lệ."
        try:
            recs = recommender.hybrid_recommendation(user_id, movie_id, num)
        except Exception as e:
            print(f"[ERROR] hybrid_recommendation exception: {e}")
            error = "Không thể lấy gợi ý. Vui lòng thử lại."

    # 5. Gán poster_url cho mọi phim trong recs nếu thiếu
    poster_cache = {}
    for lst in recs.values():
        for m in lst:
            t = m.get('title','').strip()
            if not t:
                continue
            if t in poster_cache:
                m['poster_url'] = poster_cache[t]
            else:
                try:
                    tmdb_search = recommender.search_movie_tmdb(t, max_pages=1)
                    p = tmdb_search[0].get('poster_url') if tmdb_search else None
                except Exception as e:
                    print(f"[WARN] Poster fetch for '{t}' failed: {e}")
                    p = None
                poster_cache[t] = p
                m['poster_url'] = p

    # 6. Render template
    # Nếu bạn không cần dropdown base_movie_id nữa, bỏ hẳn movies=...
    return render_template(
        'recommend.html',
        recs=recs,
        error=error,
        movies=recommender.movies_df.to_dict('records')
    )
"""

#  Danh sách genres
@app.route('/genres')
def genres():
    """Hiển thị trang danh sách genres"""
    try:
        genres = recommender.get_tmdb_genres()
    except Exception as e:
        print(f"Error fetching genres: {e}")
        genres = []
        flash("Không thể tải danh sách thể loại. Vui lòng thử lại sau.", "danger")
    
    return render_template('genres.html', genres=genres)

# Danh sách phim theo genre
@app.route('/genres/<int:genre_id>')
def genre_movies(genre_id):
    """Hiển thị danh sách phim theo genre"""
    try:
        # Lấy tên genre để hiển thị tiêu đề
        all_genres = recommender.get_tmdb_genres()
        genre_name = next((g['name'] for g in all_genres if g['id'] == genre_id), 'Unknown Genre')
        
        # Lấy phim theo genre
        movies = recommender.get_tmdb_movies_by_genre(genre_id, max_pages=1)[:20]
        
        return render_template('genre_movies.html', 
                             genre_name=genre_name, 
                             movies=movies, 
                             genre_id=genre_id)
    except Exception as e:
        print(f"Error fetching movies for genre {genre_id}: {e}")
        flash("Không thể tải danh sách phim. Vui lòng thử lại sau.", "danger")
        return redirect(url_for('genres'))
    
@app.context_processor
def inject_genres():
    """
    Cho phép base.html và mọi template đều có biến all_genres
    """
    try:
        genres = recommender.get_tmdb_genres()
    except Exception as e:
        print(f"Error fetching genres: {e}")
        genres = []
    return dict(all_genres=genres)

# Truy cập http://localhost:5000/debug/db để xem thông tin database
@app.route('/debug/db')
def debug_db():
    """Route debug để kiểm tra database"""
    try:
        cursor = recommender.conn.cursor()
        
        # Kiểm tra tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        # Kiểm tra users table structure
        cursor.execute("PRAGMA table_info(users);")
        users_structure = cursor.fetchall()
        
        # Đếm số users
        cursor.execute("SELECT COUNT(*) FROM users;")
        user_count = cursor.fetchone()[0]
        
        return f"""
        <h2>Database Debug Info</h2>
        <h3>Tables:</h3>
        <ul>{''.join([f'<li>{table[0]}</li>' for table in tables])}</ul>
        
        <h3>Users table structure:</h3>
        <ul>{''.join([f'<li>{col}</li>' for col in users_structure])}</ul>
        
        <h3>User count:</h3>
        <p>{user_count}</p>
        
        <p><a href="/">Back to home</a></p>
        """
    except Exception as e:
        return f"Database error: {e}"

#Truy cập http://localhost:5000/debug/test-register để test đăng ký thủ công
@app.route('/debug/test-register')
def test_register():
    """Test registration manually"""
    try:
        from werkzeug.security import generate_password_hash
        
        cursor = recommender.conn.cursor()
        username = "testuser123"
        email = "test123@example.com"
        password = "testpass123"
        hashed = generate_password_hash(password)
        
        # Xóa user test nếu tồn tại
        cursor.execute("DELETE FROM users WHERE username = ? OR email = ?", (username, email))
        
        # Insert user test
        cursor.execute("""
            INSERT INTO users (username, email, password, created_at)
            VALUES (?, ?, ?, datetime('now'))
        """, (username, email, hashed))
        
        recommender.conn.commit()
        
        return f"Test user created successfully: {username} / {email}"
        
    except Exception as e:
        return f"Test registration failed: {e}"

# Route để xóa database và tạo lại
@app.route('/debug/reset-db')
def reset_database():
    """Reset database - CHỈ DÙNG KHI DEVELOPMENT"""
    try:
        cursor = recommender.conn.cursor()
        
        # Drop và tạo lại tables
        cursor.execute("DROP TABLE IF EXISTS users")
        cursor.execute("DROP TABLE IF EXISTS user_ratings") 
        cursor.execute("DROP TABLE IF EXISTS movies")
        
        # Tạo lại database
        recommender.setup_database()
        recommender.load_sample_data()
        
        return "Database reset successfully! <a href='/'>Go home</a>"
        
    except Exception as e:
        return f"Database reset failed: {e}"
    

@app.route('/api/chatbot', methods=['POST'])
def chatbot_api():
    """API endpoint để xử lý yêu cầu chatbot"""
    try:
        movie_name = request.form.get('movie_name', '').strip()
        if not movie_name:
            return jsonify(success=False, error="Bạn chưa nhập tên phim."), 400
        
        # Tìm ID phim trên TMDB
        tmdb_list = recommender.search_movie_tmdb(movie_name, max_pages=1)
        if not tmdb_list:
            return jsonify(success=False, error=f"Không tìm thấy phim '{movie_name}'."), 200
        
        tmdb_id = tmdb_list[0]['id']
        recs = recommender.get_tmdb_recommendations(tmdb_id, max_pages=1)
        
        if not recs:
            return jsonify(success=False, error="TMDB không có gợi ý nào cho phim này."), 200
        
        # Giới hạn số lượng gợi ý để tránh spam chat
        recs = recs[:5]
        
        return jsonify(success=True, recs=recs)
        
    except Exception as e:
        print(f"Chatbot API error: {e}")
        return jsonify(success=False, error="Đã xảy ra lỗi khi xử lý yêu cầu."), 500
    
@app.route('/chatbot')
def chatbot():
    """Trang chatbot để người dùng chat với hệ thống gợi ý phim"""
    return render_template('chat.html')
# ----------------------------------------
# 7. Chạy Flask
# ----------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
