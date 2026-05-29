// ─── State ─────────────────────────────────────────────────────────────────────

const state = {
  history: [],        // array of movie IDs (most recent first)
  allMovies: [],
  loading: false,
};

// ─── DOM helpers ───────────────────────────────────────────────────────────────

const $ = id => document.getElementById(id);
const el = (tag, cls, html) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html) e.innerHTML = html;
  return e;
};

// ─── Toast ─────────────────────────────────────────────────────────────────────

function showToast(msg) {
  const toast = $("toast");
  toast.textContent = msg;
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 3000);
}

// ─── Movie card ────────────────────────────────────────────────────────────────

function createMovieCard(movie) {
  const watched = state.history.includes(movie.id);
  const card = el("div", `movie-card${watched ? " watched" : ""}`);
  card.dataset.id = movie.id;

  card.innerHTML = `
    <div class="card-poster">
      <img src="${API.posterUrl(movie.poster)}" alt="${movie.title}" loading="lazy"
           onerror="this.src='https://via.placeholder.com/300x450/1a1a2e/e50914?text=No+Poster'">
      ${watched ? '<div class="watched-badge">✓ Visto</div>' : ""}
    </div>
    <div class="card-info">
      <h4>${movie.title}</h4>
      <div class="card-meta">
        <span class="year">${movie.year}</span>
        <span class="rating">★ ${movie.rating}</span>
      </div>
      <div class="card-genres">${movie.genres.slice(0, 2).join(" · ")}</div>
      <button class="btn-watch ${watched ? "btn-unwatch" : ""}" data-id="${movie.id}">
        ${watched ? "Quitar de vistos" : "▶ Marcar como visto"}
      </button>
    </div>
  `;

  card.querySelector(".btn-watch").addEventListener("click", e => {
    e.stopPropagation();
    toggleWatched(movie.id);
  });

  card.addEventListener("click", () => openModal(movie));
  return card;
}

// ─── Row builder ───────────────────────────────────────────────────────────────

function createRow(title, movies, tag = "") {
  if (!movies || movies.length === 0) return null;

  const section = el("section", "row-section");
  section.innerHTML = `
    <div class="row-header">
      <h2 class="row-title">${title} ${tag ? `<span class="row-tag">${tag}</span>` : ""}</h2>
    </div>
    <div class="row-scroll-wrap">
      <button class="scroll-btn scroll-left" aria-label="anterior">&#8249;</button>
      <div class="movies-row"></div>
      <button class="scroll-btn scroll-right" aria-label="siguiente">&#8250;</button>
    </div>
  `;

  const row = section.querySelector(".movies-row");
  movies.forEach(m => row.appendChild(createMovieCard(m)));

  section.querySelector(".scroll-left").addEventListener("click", () => {
    row.scrollBy({ left: -row.clientWidth * 0.75, behavior: "smooth" });
  });
  section.querySelector(".scroll-right").addEventListener("click", () => {
    row.scrollBy({ left: row.clientWidth * 0.75, behavior: "smooth" });
  });

  // Hide arrows when not needed
  const updateArrows = () => {
    section.querySelector(".scroll-left").style.display  = row.scrollLeft > 0 ? "" : "none";
    section.querySelector(".scroll-right").style.display = row.scrollLeft < row.scrollWidth - row.clientWidth - 1 ? "" : "none";
  };
  row.addEventListener("scroll", updateArrows, { passive: true });
  setTimeout(updateArrows, 100);

  return section;
}

// ─── Hero banner ───────────────────────────────────────────────────────────────

function renderHero(movie) {
  const hero = $("hero");
  hero.style.backgroundImage = `url(${API.posterUrl(movie.poster, true)})`;
  hero.innerHTML = `
    <div class="hero-gradient"></div>
    <div class="hero-content">
      <h1 class="hero-title">${movie.title}</h1>
      <div class="hero-meta">
        <span class="hero-year">${movie.year}</span>
        <span class="hero-rating">★ ${movie.rating}</span>
        <span class="hero-genres">${movie.genres.join(" · ")}</span>
      </div>
      <div class="hero-actions">
        <button class="btn-hero btn-hero-watch" id="hero-watch-btn" data-id="${movie.id}">
          ▶ Marcar como visto
        </button>
        <button class="btn-hero btn-hero-info" id="hero-info-btn" data-id="${movie.id}">
          ℹ Más info
        </button>
      </div>
    </div>
  `;

  $("hero-watch-btn").addEventListener("click", () => toggleWatched(movie.id));
  $("hero-info-btn").addEventListener("click", () => openModal(movie));
}

// ─── Modal ─────────────────────────────────────────────────────────────────────


function openModal(movie) {
  const watched = state.history.includes(movie.id);
  const modal = $("modal");
  modal.querySelector(".modal-backdrop").innerHTML = `
    <div class="modal-content modal-content-wide">
      <button class="modal-close" id="modal-close">✕</button>
      <div class="modal-hero" style="background-image:url(${API.posterUrl(movie.poster, true)})">
        <div class="modal-hero-gradient"></div>
        <div class="modal-hero-info">
          <h2>${movie.title}</h2>
          <div class="modal-meta">
            <span>${movie.year}</span>
            <span>★ ${movie.rating}</span>
            <span>${movie.genres.join(" · ")}</span>
          </div>
          <div class="modal-hero-actions">
            <button class="btn-modal-watch ${watched ? "btn-unwatch" : ""}" id="modal-watch-btn">
              ${watched ? "✓ Ya visto — Quitar" : "▶ Marcar como visto"}
            </button>
            <button class="btn-modal-similar" id="modal-similar-btn">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
                <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
              </svg>
              Buscar similares
            </button>
          </div>
        </div>
      </div>
      <div id="modal-similar-section" class="modal-similar-section" hidden>
        <div id="modal-similar-loading" class="modal-similar-loading" hidden>
          <div class="spinner"></div>
          <p>Buscando películas similares…</p>
        </div>
        <div id="modal-similar-error" class="modal-similar-error" hidden></div>
        <div id="modal-similar-results" hidden>
          <div class="ms-tabs" id="modal-ms-tabs"></div>
          <div id="modal-ms-panels"></div>
        </div>
      </div>
    </div>
  `;

  modal.style.display = "flex";
  document.body.style.overflow = "hidden";

  $("modal-close").addEventListener("click", closeModal);
  $("modal-watch-btn").addEventListener("click", () => {
    toggleWatched(movie.id);
    closeModal();
  });

  // Similar search
  $("modal-similar-btn").addEventListener("click", async () => {
    const section = $("modal-similar-section");
    const loading = $("modal-similar-loading");
    const errorEl = $("modal-similar-error");
    const results = $("modal-similar-results");
    const btn     = $("modal-similar-btn");

    section.hidden = false;
    loading.hidden = false;
    results.hidden = true;
    errorEl.hidden = true;
    btn.disabled = true;

    try {
      const { models } = await API.recommendByUrl(API.posterUrl(movie.poster), 10);

      const tabsContainer   = $("modal-ms-tabs");
      const panelsContainer = $("modal-ms-panels");
      tabsContainer.innerHTML   = "";
      panelsContainer.innerHTML = "";

      models.forEach((model, i) => {
        const tab = el("button", `ms-tab${i === 0 ? " ms-tab-active" : ""}`);
        tab.dataset.msTab = model.key;
        tab.textContent   = model.label;
        tabsContainer.appendChild(tab);

        const panel = el("div", `ms-tab-panel${i > 0 ? " ms-tab-hidden" : ""}`);
        panel.id = `modal-tab-${model.key}`;
        const row = el("div", "ps-movies-row");
        panel.appendChild(row);
        panelsContainer.appendChild(panel);
        renderPosterResults(row, model.results);
      });

      // Tab switching
      tabsContainer.querySelectorAll("[data-ms-tab]").forEach(tab => {
        tab.addEventListener("click", () => {
          tabsContainer.querySelectorAll("[data-ms-tab]").forEach(t => t.classList.remove("ms-tab-active"));
          panelsContainer.querySelectorAll(".ms-tab-panel").forEach(p => p.classList.add("ms-tab-hidden"));
          tab.classList.add("ms-tab-active");
          document.getElementById(`modal-tab-${tab.dataset.msTab}`).classList.remove("ms-tab-hidden");
        });
      });

      results.hidden = false;
    } catch (err) {
      errorEl.textContent = `Error: ${err.message}`;
      errorEl.hidden = false;
    } finally {
      loading.hidden = true;
      btn.disabled = false;
    }
  });
}

function closeModal() {
  $("modal").style.display = "none";
  document.body.style.overflow = "";
}

$("modal").addEventListener("click", e => {
  if (e.target === $("modal")) closeModal();
});

// ─── Watch toggle ──────────────────────────────────────────────────────────────

async function toggleWatched(movieId) {
  const alreadyWatched = state.history.includes(movieId);

  if (alreadyWatched) {
    await API.removeFromHistory(movieId);
    state.history = state.history.filter(id => id !== movieId);
    const movie = state.allMovies.find(m => m.id === movieId);
    showToast(`"${movie?.title}" quitada del historial`);
  } else {
    await API.addToHistory(movieId);
    state.history = [movieId, ...state.history.filter(id => id !== movieId)];
    const movie = state.allMovies.find(m => m.id === movieId);
    showToast(`"${movie?.title}" marcada como vista`);
  }

  renderApp();
}

// ─── Main render ───────────────────────────────────────────────────────────────

async function renderApp() {
  const main = $("main-content");
  main.innerHTML = "";

  const watchedMovies = state.history
    .map(id => state.allMovies.find(m => m.id === id))
    .filter(Boolean);

  // Hero: most recently watched or top-rated
  const heroMovie = watchedMovies[0] || state.allMovies[0];
  renderHero(heroMovie);

  // "Continuar viendo" (history)
  if (watchedMovies.length > 0) {
    const histRow = createRow("Continuar viendo", watchedMovies.slice(0, 10));
    if (histRow) main.appendChild(histRow);
  }

  // "Recomendado para ti" — global recommendations based on full history
  const { recommendations: forYou } = await API.getRecommendations(state.history, 12);
  const forYouRow = createRow("Recomendado para ti", forYou, state.history.length > 0 ? "por portada" : "");
  if (forYouRow) main.appendChild(forYouRow);

  // "Porque viste X" — per-movie recommendation rows (up to 3 watched movies)
  for (const movie of watchedMovies.slice(0, 3)) {
    const { recommendations } = await API.getRecommendationsForMovie(movie.id, 10);
    const row = createRow(`Porque viste`, recommendations, `"${movie.title}"`);
    if (row) main.appendChild(row);
  }

  // "Populares" — top-rated across the catalog
  const popular = [...state.allMovies]
    .sort((a, b) => b.rating - a.rating)
    .slice(0, 12);
  const popularRow = createRow("Populares en VisionFlix", popular);
  if (popularRow) main.appendChild(popularRow);

  // Genre rows
  const genres = ["Action", "Drama", "Animation", "Thriller", "Science Fiction"];
  for (const genre of genres) {
    const movies = state.allMovies
      .filter(m => m.genres.includes(genre))
      .sort((a, b) => b.rating - a.rating)
      .slice(0, 10);
    if (movies.length >= 3) {
      const row = createRow(genre, movies);
      if (row) main.appendChild(row);
    }
  }
}

// ─── Search ────────────────────────────────────────────────────────────────────

function setupSearch() {
  const input = $("search-input");
  const overlay = $("search-overlay");
  const results = $("search-results");

  input.addEventListener("input", () => {
    const q = input.value.trim().toLowerCase();
    if (!q) { overlay.style.display = "none"; return; }

    const matches = state.allMovies.filter(m =>
      m.title.toLowerCase().includes(q) ||
      m.genres.some(g => g.toLowerCase().includes(q))
    ).slice(0, 8);

    if (matches.length === 0) {
      results.innerHTML = '<p class="no-results">Sin resultados</p>';
    } else {
      results.innerHTML = "";
      matches.forEach(m => {
        const item = el("div", "search-item");
        item.innerHTML = `
          <img src="${API.posterUrl(m.poster)}" alt="${m.title}" onerror="this.style.display='none'">
          <div>
            <div class="si-title">${m.title}</div>
            <div class="si-meta">${m.year} · ${m.genres.slice(0, 2).join(", ")}</div>
          </div>
        `;
        item.addEventListener("click", () => {
          input.value = "";
          overlay.style.display = "none";
          openModal(m);
        });
        results.appendChild(item);
      });
    }
    overlay.style.display = "block";
  });

  document.addEventListener("click", e => {
    if (!input.contains(e.target) && !overlay.contains(e.target)) {
      overlay.style.display = "none";
    }
  });
}

// ─── Nav scroll effect ─────────────────────────────────────────────────────────

window.addEventListener("scroll", () => {
  const nav = $("navbar");
  nav.classList.toggle("scrolled", window.scrollY > 50);
}, { passive: true });

function renderPosterResults(container, movies) {
  container.innerHTML = "";
  if (!movies || movies.length === 0) {
    container.innerHTML = '<p class="ps-empty">Sin resultados</p>';
    return;
  }
  movies.forEach(movie => {
    const card = el("div", "ps-result-card");
    const poster = movie.tmdb_poster_url ||
      `https://via.placeholder.com/160x240/1a1a2e/e50914?text=${encodeURIComponent(movie.title)}`;
    const genres = (movie.genres || "").split(", ").slice(0, 2).join(" · ");
    const pct = Math.round((movie.score || 0) * 100);
    card.innerHTML = `
      <div class="ps-card-poster">
        <img src="${poster}" alt="${movie.title}" loading="lazy"
             onerror="this.src='https://via.placeholder.com/160x240/1a1a2e/e50914?text=No+Poster'">
        <div class="ps-score-badge">${pct}%</div>
      </div>
      <div class="ps-card-info">
        <div class="ps-card-title">${movie.title}</div>
        <div class="ps-card-genres">${genres}</div>
      </div>
    `;
    container.appendChild(card);
  });
}

// ─── Init ──────────────────────────────────────────────────────────────────────

async function init() {
  $("loading-screen").style.display = "flex";

  await _sleep(5000); // ← duración mínima del spinner, independiente del mock

  const [{ movies }, { history }] = await Promise.all([
    API.getMovies(),
    API.getHistory(),
  ]);

  state.allMovies = movies;
  state.history = history;

  $("loading-screen").style.display = "none";

  setupSearch();
  await renderApp();
}

document.addEventListener("DOMContentLoaded", init);
