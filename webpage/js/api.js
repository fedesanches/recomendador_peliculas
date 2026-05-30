/**
 * VisionFlix API Client
 *
 * To connect a real backend, set USE_MOCK = false and configure API_BASE_URL.
 * The real /recommend endpoint must accept { watched_ids: number[], limit: number }
 * and return { recommendations: Movie[] } — same shape as the mock.
 */

const API_CONFIG = {
  USE_MOCK: true,
  BASE_URL: "",   // mismo origen — funciona local (:7860) y en HF Spaces
  POSTER_BASE: "https://image.tmdb.org/t/p/w500",
  POSTER_HERO: "https://image.tmdb.org/t/p/w1280",
  MOCK_DELAY_MS: 300,
};

// ─── Mock persistence (localStorage) ──────────────────────────────────────────

function _loadHistory() {
  try {
    return JSON.parse(localStorage.getItem("visionflix_history") || "[]");
  } catch {
    return [];
  }
}

function _saveHistory(ids) {
  localStorage.setItem("visionflix_history", JSON.stringify(ids));
}

// ─── Mock implementation ───────────────────────────────────────────────────────

function _sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

function _mockRecommend(watchedIds, limit) {
  if (watchedIds.length === 0) {
    // No history → return top-rated movies
    return [...MOCK_MOVIES]
      .sort((a, b) => b.rating - a.rating)
      .filter(m => !watchedIds.includes(m.id))
      .slice(0, limit);
  }

  const watchedSet = new Set(watchedIds);

  // Collect genres from watched movies
  const genreScore = {};
  for (const id of watchedIds) {
    const movie = MOCK_MOVIES.find(m => m.id === id);
    if (!movie) continue;
    for (const g of movie.genres) {
      genreScore[g] = (genreScore[g] || 0) + 1;
    }
  }

  // Score unwatched movies by genre overlap (simulates poster-embedding similarity)
  const candidates = MOCK_MOVIES
    .filter(m => !watchedSet.has(m.id))
    .map(m => {
      const score = m.genres.reduce((s, g) => s + (genreScore[g] || 0), 0);
      // Add small rating boost to break ties
      return { ...m, _score: score + m.rating * 0.05 };
    })
    .sort((a, b) => b._score - a._score);

  return candidates.slice(0, limit);
}

function _mockRecommendForMovie(movieId, limit) {
  const source = MOCK_MOVIES.find(m => m.id === movieId);
  if (!source) return [];

  return MOCK_MOVIES
    .filter(m => m.id !== movieId)
    .map(m => {
      const overlap = m.genres.filter(g => source.genres.includes(g)).length;
      return { ...m, _score: overlap * 2 + m.rating * 0.1 };
    })
    .sort((a, b) => b._score - a._score)
    .slice(0, limit);
}

// ─── Public API ────────────────────────────────────────────────────────────────

const API = {
  /**
   * Returns all movies in the catalog.
   * GET /api/movies → { movies: Movie[] }
   */
  async getMovies() {
    if (API_CONFIG.USE_MOCK) {
      await _sleep(API_CONFIG.MOCK_DELAY_MS);
      return { movies: MOCK_MOVIES };
    }
    const res = await fetch(`${API_CONFIG.BASE_URL}/api/movies`);
    return res.json();
  },

  /**
   * Returns a single movie by id.
   * GET /api/movies/:id → Movie
   */
  async getMovie(id) {
    if (API_CONFIG.USE_MOCK) {
      await _sleep(API_CONFIG.MOCK_DELAY_MS);
      const movie = MOCK_MOVIES.find(m => m.id === id) || null;
      return { movie };
    }
    const res = await fetch(`${API_CONFIG.BASE_URL}/api/movies/${id}`);
    return res.json();
  },

  /**
   * Returns recommendations based on a list of watched movie IDs.
   * This is the main endpoint for the poster-based recommender.
   * POST /api/recommend → body: { watched_ids: number[], limit: number }
   *                     → { recommendations: Movie[] }
   */
  async getRecommendations(watchedIds, limit = 12) {
    if (API_CONFIG.USE_MOCK) {
      await _sleep(API_CONFIG.MOCK_DELAY_MS);
      return { recommendations: _mockRecommend(watchedIds, limit) };
    }
    const res = await fetch(`${API_CONFIG.BASE_URL}/api/recommend`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ watched_ids: watchedIds, limit }),
    });
    return res.json();
  },

  /**
   * Returns recommendations similar to a specific movie.
   * POST /api/recommend/movie → body: { movie_id: number, limit: number }
   *                           → { recommendations: Movie[] }
   */
  async getRecommendationsForMovie(movieId, limit = 10) {
    if (API_CONFIG.USE_MOCK) {
      await _sleep(API_CONFIG.MOCK_DELAY_MS);
      return { recommendations: _mockRecommendForMovie(movieId, limit) };
    }
    const res = await fetch(`${API_CONFIG.BASE_URL}/api/recommend/movie`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ movie_id: movieId, limit }),
    });
    return res.json();
  },

  /**
   * Returns the user's watch history.
   * GET /api/user/history → { history: number[] }
   */
  async getHistory() {
    if (API_CONFIG.USE_MOCK) {
      await _sleep(API_CONFIG.MOCK_DELAY_MS / 2);
      return { history: _loadHistory() };
    }
    const res = await fetch(`${API_CONFIG.BASE_URL}/api/user/history`);
    return res.json();
  },

  /**
   * Marks a movie as watched.
   * POST /api/user/history → body: { movie_id: number } → { success: boolean }
   */
  async addToHistory(movieId) {
    if (API_CONFIG.USE_MOCK) {
      await _sleep(API_CONFIG.MOCK_DELAY_MS / 2);
      const history = _loadHistory();
      if (!history.includes(movieId)) {
        history.unshift(movieId);
        _saveHistory(history);
      }
      return { success: true };
    }
    const res = await fetch(`${API_CONFIG.BASE_URL}/api/user/history`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ movie_id: movieId }),
    });
    return res.json();
  },

  /**
   * Removes a movie from watch history.
   * DELETE /api/user/history/:id → { success: boolean }
   */
  async removeFromHistory(movieId) {
    if (API_CONFIG.USE_MOCK) {
      await _sleep(API_CONFIG.MOCK_DELAY_MS / 2);
      const history = _loadHistory().filter(id => id !== movieId);
      _saveHistory(history);
      return { success: true };
    }
    const res = await fetch(`${API_CONFIG.BASE_URL}/api/user/history/${movieId}`, {
      method: "DELETE",
    });
    return res.json();
  },

  /**
   * Sends a TMDB poster URL to the backend; the server downloads the image
   * and queries both indices in parallel.
   * POST /recommend/url?top_k=N&combined=false|true  body: { url }
   * Returns { image: Movie[], combined: Movie[] }
   */
  async recommendByUrl(url, topK = 10) {
    const res = await fetch(
      `${API_CONFIG.BASE_URL}/recommend/url/all?top_k=${topK}`,
      {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ url }),
      }
    );
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Error al consultar el recomendador");
    }
    return res.json(); // { image: [...], combined: [...] }
  },

  /**
   * Uploads an image file and returns recommendations from all available models.
   * POST /recommend/image?top_k=N  body: multipart/form-data { file }
   * Returns { models: [{ key, label, results }] }
   */
  async recommendByImage(file, topK = 10) {
    if (API_CONFIG.USE_MOCK) {
      await _sleep(1200);
      const shuffle = arr => [...arr].sort(() => Math.random() - 0.5);
      const toResults = movies => shuffle(movies).slice(0, topK).map(m => ({
        title: m.title,
        score: 0.55 + Math.random() * 0.45,
        tmdb_poster_url: API_CONFIG.POSTER_BASE + m.poster,
        genres: m.genres.join(", "),
      }));
      return {
        models: [
          { key: "clip-image",         label: "CLIP — Imagen",              results: toResults(MOCK_MOVIES) },
          { key: "clip-combined",      label: "CLIP — Imagen + Texto",      results: toResults(MOCK_MOVIES) },
          { key: "notextimg",          label: "CLIP — Sin texto",           results: toResults(MOCK_MOVIES) },
          { key: "notextimg-combined", label: "CLIP — Sin texto + Texto",   results: toResults(MOCK_MOVIES) },
          { key: "siglip",             label: "SigLIP",                     results: toResults(MOCK_MOVIES) },
          { key: "dinov2",             label: "DINOv2",                     results: toResults(MOCK_MOVIES) },
        ],
      };
    }

    const VARIANTS = [
      { key: "clip-image",         label: "CLIP — Imagen",              params: `top_k=${topK}` },
      { key: "clip-combined",      label: "CLIP — Imagen + Texto",      params: `top_k=${topK}&combined=true` },
      { key: "notextimg",          label: "CLIP — Sin texto",           params: `top_k=${topK}&model=notextimg` },
      { key: "notextimg-combined", label: "CLIP — Sin texto + Texto",   params: `top_k=${topK}&model=notextimg_combined` },
      { key: "siglip",             label: "SigLIP",                     params: `top_k=${topK}&model=siglip` },
      { key: "dinov2",             label: "DINOv2",                     params: `top_k=${topK}&model=dinov2` },
    ];

    const settled = await Promise.allSettled(
      VARIANTS.map(v => {
        const fd = new FormData();
        fd.append("file", file);
        return fetch(`${API_CONFIG.BASE_URL}/recommend/image?${v.params}`, {
          method: "POST",
          body: fd,
        })
          .then(r => r.ok ? r.json() : Promise.reject(new Error(r.status)))
          .then(data => ({ ...v, results: data.results || [] }));
      })
    );

    const models = settled
      .filter(r => r.status === "fulfilled")
      .map(r => r.value);

    if (models.length === 0) throw new Error("No hay modelos disponibles o el servidor no está activo");
    return { models };
  },

  posterUrl(path, hero = false) {
    if (!path) return "https://via.placeholder.com/500x750?text=No+Poster";
    const base = hero ? API_CONFIG.POSTER_HERO : API_CONFIG.POSTER_BASE;
    return `${base}${path}`;
  },
};
