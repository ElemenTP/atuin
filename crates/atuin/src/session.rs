//! In-process Atuin session.
//!
//! This module is the in-process equivalent of zoxide's `src/session.rs`: it
//! does not reimplement history recording or search. Instead it holds the
//! long-lived SQLite/record-store handles and runtime needed by an embedded
//! shell integration, then delegates the actual command behaviour to the same
//! upstream client-command code that the `atuin` binary uses:
//!
//! * `history start` → [`crate::command::client::history::handle_start_with_cwd`]
//! * `history end` → [`crate::command::client::history::handle_end`]
//! * non-interactive search → [`crate::command::client::search::run_non_interactive_with_context`]
//! * interactive search → [`crate::command::client::search::interactive::history`]
//!
//! [`Session::search`] dispatches on [`SearchMode`] exactly like the upstream
//! `atuin search` command (`--interactive`), and the `search_prefix` /
//! `search_interactive` helpers are thin convenience wrappers around it.
//!
//! Project-specific concerns that do not exist in the CLI (multi-thread
//! fire-and-forget `history end`, pre-`dlclose` resource shutdown, and the
//! custom terminal input that avoids signal-hook callbacks) are injected here
//! behind the `in-process` feature.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use atuin_client::database::{OptFilters, Sqlite, query_context};
use atuin_client::history::store::HistoryStore;
use atuin_client::history::{AuthorKind, AuthorPattern, History, HistoryId};
use atuin_client::record::sqlite_store::SqliteStore;
use atuin_client::settings::{FilterMode, KeymapMode, RequestedSearchMode, Settings};
use atuin_common::encryption::paseto_v4;
use atuin_common::filter::OrFilter;
use eyre::{Result, eyre};
use tokio::runtime::Runtime;
use tokio::sync::Notify;

use crate::command::client::history::{handle_end, handle_start_with_cwd};
use crate::command::client::search::interactive;
use crate::command::client::search::run_non_interactive_with_context;
use atuin_client::theme::ThemeManager;

/// Which search UI [`Session::search`] should run.
///
/// This mirrors the upstream `atuin search --interactive` flag: one entry point
/// selects the search engine, then either returns matching history entries or
/// opens the full-screen TUI.
#[derive(Clone, Debug, Default)]
pub enum SearchMode {
    /// Non-interactive query engine: returns matching [`History`] entries.
    #[default]
    NonInteractive,
    /// In-process interactive TUI: returns the selected command, if any.
    Interactive,
}

/// Result of [`Session::search`].
#[derive(Debug)]
pub enum SearchResult {
    /// Non-interactive matches (newest first unless `reverse` is set).
    Entries(Vec<History>),
    /// Interactive selection. `None` means the user cancelled and the shell
    /// should leave its buffer unchanged.
    Interactive(Option<String>),
}

impl SearchResult {
    /// Extract non-interactive entries, or fail if the search was interactive.
    pub fn into_entries(self) -> Result<Vec<History>> {
        match self {
            Self::Entries(entries) => Ok(entries),
            Self::Interactive(_) => Err(eyre!("expected a non-interactive search result")),
        }
    }

    /// Extract an interactive selection, or fail if the search was
    /// non-interactive.
    pub fn into_interactive(self) -> Result<Option<String>> {
        match self {
            Self::Interactive(selection) => Ok(selection),
            Self::Entries(_) => Err(eyre!("expected an interactive search result")),
        }
    }
}

/// Options for one [`Session::search`] call.
///
/// The field set mirrors the upstream `atuin search` command's filtering and
/// search-mode flags. A `None`/empty value means "use the configured default",
/// matching the CLI when the corresponding flag is omitted. Non-interactive
/// fields (`cwd`, `limit`, `authors`, ...) are ignored by the interactive TUI,
/// exactly as in the CLI.
#[derive(Clone, Debug, Default)]
pub struct SearchOptions {
    /// Search text. Upstream treats multiple query words as a single
    /// space-joined query string for non-interactive searches; the interactive
    /// TUI receives it as the initial query.
    pub query: String,
    /// Non-interactive query engine or interactive TUI.
    pub mode: SearchMode,
    /// Search engine mode (prefix/full-text/fuzzy/daemon-fuzzy).
    pub search_mode: Option<RequestedSearchMode>,
    /// Result filter mode (global/host/session/directory/workspace/...).
    pub filter_mode: Option<FilterMode>,
    pub cwd: Option<String>,
    pub exclude_cwd: Option<String>,
    pub exit: Option<i64>,
    pub exclude_exit: Option<i64>,
    pub before: Option<String>,
    pub after: Option<String>,
    pub limit: Option<i64>,
    pub offset: Option<i64>,
    pub reverse: bool,
    pub include_duplicates: bool,
    /// Author filters. Literal names, `$all-user`, and `$all-agent` are all
    /// accepted by `AuthorPattern::from(&str)`.
    pub authors: Vec<AuthorPattern>,
    /// Shell filters. An empty string matches commands with no recorded shell.
    pub shells: Vec<String>,
    /// Interactive-only: mirror `atuin search --shell-up-key-binding` (used by
    /// the shell's UpArrow widget).
    pub shell_up_key_binding: bool,
    /// Interactive-only: mirror `atuin search --keymap-mode` (used by the
    /// zsh/pwsh vi widgets).
    pub keymap_mode: Option<KeymapMode>,
}

impl SearchOptions {
    /// A default (non-interactive) search for `query`.
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            ..Default::default()
        }
    }

    /// An interactive TUI search prefilled with `query`.
    pub fn interactive(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            mode: SearchMode::Interactive,
            ..Default::default()
        }
    }

    /// An interactive TUI search opened from a shell UpArrow binding
    /// (`atuin search -i --shell-up-key-binding`).
    pub fn interactive_up(query: impl Into<String>) -> Self {
        Self {
            shell_up_key_binding: true,
            ..Self::interactive(query)
        }
    }

    /// The shell autosuggest query: official
    /// `atuin search --cmd-only --author '$all-user' --limit N --search-mode prefix`.
    pub fn prefix(query: impl Into<String>, limit: usize) -> Self {
        Self {
            query: query.into(),
            search_mode: Some(RequestedSearchMode::Prefix),
            limit: Some(i64::try_from(limit).unwrap_or(i64::MAX)),
            authors: vec![AuthorPattern::AllUser],
            ..Default::default()
        }
    }

    /// Mirror `atuin search --shell-up-key-binding`.
    pub fn with_shell_up_key_binding(mut self, enabled: bool) -> Self {
        self.shell_up_key_binding = enabled;
        self
    }

    /// Mirror `atuin search --keymap-mode` for the vi widgets.
    pub fn with_keymap_mode(mut self, keymap_mode: KeymapMode) -> Self {
        self.keymap_mode = Some(keymap_mode);
        self
    }
}

/// Tracks the number of fire-and-forget `history_end` tasks still running.
///
/// This deliberately stores only a counter/notifier instead of a `JoinHandle`
/// per command: an interactive shell can issue a very large number of
/// `history_end` calls during a long session, and the tasks are short-lived
/// DB writes. `Drop` only needs to wait until the counter reaches zero before
/// closing the SQLite pools.
struct InFlightHistoryEnd {
    count: AtomicUsize,
    notify: Notify,
}

struct InFlightHistoryEndGuard {
    in_flight: Arc<InFlightHistoryEnd>,
}

impl Drop for InFlightHistoryEndGuard {
    fn drop(&mut self) {
        if self.in_flight.count.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.in_flight.notify.notify_one();
        }
    }
}

/// Per-session operation counters.
///
/// Mirrors the starship in-process session stats: the shell integration can
/// show what the native session has actually done instead of paying for a
/// child process just to inspect it.
#[derive(Debug, Clone, Default)]
pub struct SessionStats {
    /// `history_start` calls that reached the session.
    pub history_starts: u64,
    /// Blocking `history_end` calls.
    pub history_ends_sync: u64,
    /// Fire-and-forget `history_end` calls.
    pub history_ends_async: u64,
    /// Every `search` dispatch (interactive and non-interactive).
    pub search_calls: u64,
    /// `search_prefix` shortcut calls (also counted in `search_calls`).
    pub search_prefix_calls: u64,
    /// Interactive TUI sessions opened.
    pub interactive_search_calls: u64,
    /// Interactive searches that accepted a history entry.
    pub interactive_selections: u64,
    /// Interactive searches the user cancelled.
    pub interactive_cancels: u64,
    /// Fire-and-forget `history_end` writes still pending.
    pub in_flight_history_ends: u64,
    /// Seconds since this session was created.
    pub uptime_secs: u64,
}

/// Atomic counters backing [`SessionStats`].
#[derive(Debug, Default)]
struct SessionStatsCounters {
    history_starts: AtomicU64,
    history_ends_sync: AtomicU64,
    history_ends_async: AtomicU64,
    search_calls: AtomicU64,
    search_prefix_calls: AtomicU64,
    interactive_search_calls: AtomicU64,
    interactive_selections: AtomicU64,
    interactive_cancels: AtomicU64,
}

/// A persistent in-process atuin session.
pub struct Session {
    settings: Arc<Settings>,
    history_db: Sqlite,
    history_store: HistoryStore,
    session_id: String,
    runtime: Option<Runtime>,
    in_flight_history_end: Arc<InFlightHistoryEnd>,
    stats: SessionStatsCounters,
    created_at: std::time::Instant,
}

impl Session {
    /// Create a session using the data directory resolved from the user's
    /// Atuin configuration (`ATUIN_DATA_DIR`, XDG, or `data_dir` in
    /// config.toml), exactly like the official CLI.
    pub fn new() -> Result<Self> {
        Self::new_with_datadir(None)
    }

    /// Create a session.
    ///
    /// `Some(data_dir)` uses `data_dir` as the Atuin data directory.
    /// `None` follows the fully-resolved settings (`ATUIN_DATA_DIR`, XDG, or
    /// `data_dir` in config.toml) instead of second-guessing the config layer.
    ///
    /// Prefer [`Session::new`]; the explicit data directory exists for tests
    /// that need to bypass the settings layer.
    pub fn new_with_datadir(data_dir: Option<&std::path::Path>) -> Result<Self> {
        // A multi-thread runtime is used deliberately. The shell integration
        // wants `atuin_history_end` to be genuinely fire-and-forget: the
        // builtin schedules the database update on a tokio worker thread and
        // returns immediately, exactly like the official
        // `(atuin history end ... &)` shell background job. A current-thread
        // runtime can only poll spawned tasks while `block_on` is on the
        // stack, so an async end would not make progress until the next
        // in-process FFI call.
        //
        // Worker-thread lifecycle is scoped to this Session: `shutdown()`
        // calls `Runtime::shutdown_timeout` and blocks until the workers have
        // exited BEFORE `dlclose` can unmap the shared library.
        let runtime =
            tokio::runtime::Builder::new_multi_thread().worker_threads(8).enable_all().build()?;

        // Settings::new() (not builder().try_deserialize()) also registers the
        // lazy meta-store config needed by Settings::host_id() below.
        let settings = Arc::new(Settings::new()?);

        let (db_path, record_store_path, key_path) = match data_dir {
            Some(data_dir) => {
                (data_dir.join("history.db"), data_dir.join("records.db"), data_dir.join("key"))
            }
            None => (
                settings.db_path.clone(),
                settings.record_store_path.clone(),
                settings.key_path.clone(),
            ),
        };

        let (history_db, history_store, session_id) = runtime.block_on(async {
            let timeout = Duration::try_from_secs_f64(settings.local_timeout)?;

            let history_db = Sqlite::new(&db_path, timeout).await?;
            let record_store = SqliteStore::new(record_store_path, timeout).await?;

            // Load or create encryption key.
            let encryption_key = paseto_v4::Key::try_load_or_generate(&key_path)?;

            let history_store =
                HistoryStore::new(record_store, Settings::host_id().await?, encryption_key);

            let session_id = atuin_common::utils::uuid_v7().as_simple().to_string();

            Ok::<_, eyre::Report>((history_db, history_store, session_id))
        })?;

        Ok(Self {
            settings,
            history_db,
            history_store,
            session_id,
            runtime: Some(runtime),
            in_flight_history_end: Arc::new(InFlightHistoryEnd {
                count: AtomicUsize::new(0),
                notify: Notify::new(),
            }),
            stats: SessionStatsCounters::default(),
            created_at: std::time::Instant::now(),
        })
    }

    /// Snapshot the session's operation counters.
    #[must_use]
    pub fn stats(&self) -> SessionStats {
        SessionStats {
            history_starts: self.stats.history_starts.load(Ordering::Relaxed),
            history_ends_sync: self.stats.history_ends_sync.load(Ordering::Relaxed),
            history_ends_async: self.stats.history_ends_async.load(Ordering::Relaxed),
            search_calls: self.stats.search_calls.load(Ordering::Relaxed),
            search_prefix_calls: self.stats.search_prefix_calls.load(Ordering::Relaxed),
            interactive_search_calls: self.stats.interactive_search_calls.load(Ordering::Relaxed),
            interactive_selections: self.stats.interactive_selections.load(Ordering::Relaxed),
            interactive_cancels: self.stats.interactive_cancels.load(Ordering::Relaxed),
            in_flight_history_ends: self.in_flight_history_end.count.load(Ordering::Relaxed) as u64,
            uptime_secs: self.created_at.elapsed().as_secs(),
        }
    }

    fn rt(&self) -> &Runtime {
        self.runtime.as_ref().expect("runtime already shut down")
    }

    /// Record a command start. Mirrors the official `atuin history start`
    /// implementation, including command normalization, author probing,
    /// NUL-byte rejection and silent DB-error handling.
    pub fn history_start(
        &self,
        command: &str,
        cwd: &str,
        author: Option<&str>,
        author_kind: Option<AuthorKind>,
        intent: Option<&str>,
    ) -> Result<Option<String>> {
        self.stats.history_starts.fetch_add(1, Ordering::Relaxed);

        // Upstream started returning a typed `HistoryId` (v18.22.0). The FFI
        // boundary (and the shell-side protocol) still exchanges the simple
        // hyphen-less UUID string produced by `HistoryId::to_string()`.
        let id = self.rt().block_on(handle_start_with_cwd(
            &self.history_db,
            &self.settings,
            command,
            cwd,
            author,
            author_kind,
            intent,
        ))?;
        Ok(id.map(|id| id.to_string()))
    }

    /// Blocking history end. Mirrors the official `atuin history end`
    /// implementation (duration is inferred from the start timestamp when no
    /// duration is supplied).
    pub fn history_end(&self, id: &str, exit: i64, duration_ns: u64) -> Result<()> {
        self.stats.history_ends_sync.fetch_add(1, Ordering::Relaxed);

        let id = id.parse::<HistoryId>()?;
        let duration = (duration_ns > 0).then_some(duration_ns);
        self.rt().block_on(handle_end(
            &self.history_db,
            self.history_store.store.clone(),
            self.history_store.clone(),
            &self.settings,
            id,
            exit,
            duration,
        ))
    }

    /// Fire-and-forget history end. Spawns on the multi-thread tokio runtime
    /// and returns immediately. Errors are logged but not surfaced to the
    /// caller — matching the official `(atuin history end ... &)` behavior.
    pub fn history_end_async(&self, id: String, exit: i64, duration_ns: u64) {
        self.stats.history_ends_async.fetch_add(1, Ordering::Relaxed);

        let id = match id.parse::<HistoryId>() {
            Ok(id) => id,
            Err(e) => {
                tracing::debug!("async history_end got invalid history id: {e}");
                return;
            }
        };
        let duration = (duration_ns > 0).then_some(duration_ns);
        let db = self.history_db.clone();
        let store = self.history_store.store.clone();
        let history_store = self.history_store.clone();
        let settings = self.settings.clone();

        let in_flight = Arc::clone(&self.in_flight_history_end);
        in_flight.count.fetch_add(1, Ordering::AcqRel);
        let guard = InFlightHistoryEndGuard { in_flight };

        self.rt().spawn(async move {
            let _guard = guard;
            if let Err(e) =
                handle_end(&db, store, history_store, &settings, id, exit, duration).await
            {
                tracing::debug!("async history_end failed: {e}");
            }
        });
    }

    /// Load one history entry by ID.
    ///
    /// Useful for debug/tests and for shell integrations that need to verify
    /// an asynchronous `history_end` has been persisted.
    pub fn get_history(&self, id: &str) -> Result<Option<History>> {
        let id = id.parse::<HistoryId>()?;
        Ok(self.rt().block_on(self.history_db.load(id))?)
    }

    /// Run a history search.
    ///
    /// This is the single entry point for both official `atuin search` modes:
    ///
    /// * [`SearchMode::NonInteractive`] uses the upstream query engine and
    ///   returns [`SearchResult::Entries`];
    /// * [`SearchMode::Interactive`] opens the upstream full-screen TUI and
    ///   returns [`SearchResult::Interactive`].
    ///
    /// Like the CLI's `Cmd::run`, the search-mode/filter-mode/keymap overrides
    /// are applied to a copy of the session settings before dispatch, and
    /// non-interactive-only fields are ignored in interactive mode.
    pub fn search(&self, options: SearchOptions) -> Result<SearchResult> {
        self.stats.search_calls.fetch_add(1, Ordering::Relaxed);

        let mut settings = (*self.settings).clone();
        if let Some(mode) = options.search_mode {
            settings.requested_search_mode = mode;
        }
        if let Some(filter_mode) = options.filter_mode {
            settings.filter_mode = Some(filter_mode);
        }
        settings.shell_up_key_binding = options.shell_up_key_binding;
        if let Some(keymap_mode) = options.keymap_mode {
            // Config beats the shell widget unless it is on "auto", matching
            // `atuin search --keymap-mode`.
            settings.keymap_mode = match settings.keymap_mode {
                KeymapMode::Auto => keymap_mode,
                configured => configured,
            };
            settings.keymap_mode_shell = keymap_mode;
        }

        match options.mode {
            SearchMode::NonInteractive => self.search_non_interactive(&settings, options),
            SearchMode::Interactive => self.search_interactive_tui(&settings, options.query),
        }
    }

    /// Non-interactive branch of [`Self::search`].
    fn search_non_interactive(
        &self,
        settings: &Settings,
        options: SearchOptions,
    ) -> Result<SearchResult> {
        // Destructure first so the non-query fields can be borrowed by
        // `OptFilters` while the query is moved into the query slice.
        let SearchOptions {
            query,
            cwd,
            exclude_cwd,
            exit,
            exclude_exit,
            before,
            after,
            limit,
            offset,
            reverse,
            include_duplicates,
            authors,
            shells,
            ..
        } = options;

        let entries = self.rt().block_on(async move {
            let authors = OrFilter::from_list(authors).unwrap_or_default();
            let shells = OrFilter::from_list(shells).unwrap_or_default();

            let context = query_context().await?;
            let opt_filter = OptFilters {
                exit,
                exclude_exit,
                only_failed: false,
                cwd: cwd.as_deref(),
                exclude_cwd: exclude_cwd.as_deref(),
                before: before.as_deref(),
                after: after.as_deref(),
                limit,
                offset,
                reverse,
                include_duplicates,
                authors: authors.as_slice_filter(),
                shells: shells.as_slice_filter(),
            };

            let query = [query];
            run_non_interactive_with_context(
                settings,
                opt_filter,
                &query,
                &self.history_db,
                context,
            )
            .await
        })?;

        Ok(SearchResult::Entries(entries))
    }

    /// Interactive branch of [`Self::search`].
    fn search_interactive_tui(&self, settings: &Settings, query: String) -> Result<SearchResult> {
        self.stats.interactive_search_calls.fetch_add(1, Ordering::Relaxed);

        let db = self.history_db.clone();
        let history_store = self.history_store.clone();
        let result = self.rt().block_on(async move {
            let mut theme_manager = ThemeManager::new(settings.theme.debug, None);
            let theme =
                theme_manager.load_theme(settings.theme.name.as_str(), settings.theme.max_depth);
            interactive::history(&[query], settings, db, &history_store, theme).await
        })?;

        if result.is_empty() {
            self.stats.interactive_cancels.fetch_add(1, Ordering::Relaxed);
            Ok(SearchResult::Interactive(None))
        } else {
            self.stats.interactive_selections.fetch_add(1, Ordering::Relaxed);
            Ok(SearchResult::Interactive(Some(result)))
        }
    }

    /// Convenience for the shell autosuggest path: official
    /// `atuin search --cmd-only --author '$all-user' --limit N --search-mode prefix`.
    pub fn search_prefix(&self, query: &str, limit: usize) -> Result<Vec<History>> {
        self.stats.search_prefix_calls.fetch_add(1, Ordering::Relaxed);

        self.search(SearchOptions::prefix(query, limit))?.into_entries()
    }

    /// Quick path for the shell's Ctrl+R widget: in-process interactive TUI.
    ///
    /// Returns `Ok(None)` when the caller should leave the shell buffer
    /// unchanged (cancel / return-original), and `Ok(Some(command))` for an
    /// accepted history entry or a returned query.
    pub fn search_interactive(&self, query: &str) -> Result<Option<String>> {
        self.search(SearchOptions::interactive(query))?.into_interactive()
    }

    /// Quick path for the shell's UpArrow widget: official
    /// `atuin search -i --shell-up-key-binding`.
    pub fn search_interactive_up(&self, query: &str) -> Result<Option<String>> {
        self.search(SearchOptions::interactive_up(query))?.into_interactive()
    }

    /// Quick path for interactive searches that need extra official flags
    /// (`--keymap-mode`, `--shell-up-key-binding`, ...).
    pub fn search_interactive_with(&self, options: SearchOptions) -> Result<Option<String>> {
        self.search(options)?.into_interactive()
    }

    pub fn session_id_str(&self) -> &str {
        &self.session_id
    }
}

impl Drop for Session {
    /// Shut down the session.
    ///
    /// All work is kept inside the session's own tokio runtime so no external
    /// executor (and no extra thread-local state from `futures::executor`) is
    /// introduced while unloading a loadable module:
    ///
    /// 1. wait for the in-flight history_end counter to reach zero;
    /// 2. close the sqlx pools while still inside the runtime;
    /// 3. shut down the runtime and wait for tokio/sqlx workers to exit.
    fn drop(&mut self) {
        // The TUI event source caches an open terminal handle in a
        // process-global `OnceLock`; release it so the fd and any buffered
        // input do not survive into the next session (or the library unload).
        crate::command::client::search::reset_tui_input();

        let Some(rt) = self.runtime.take() else {
            return;
        };

        let history_db = self.history_db.clone();
        let history_store = self.history_store.clone();
        let in_flight = Arc::clone(&self.in_flight_history_end);

        rt.block_on(async {
            // Wait for all short fire-and-forget history_end writes to finish.
            // No JoinHandle list is kept, so this is O(1) regardless of how
            // many history_end calls happened during the shell session.
            while in_flight.count.load(Ordering::Acquire) != 0 {
                in_flight.notify.notified().await;
            }

            // sqlx SQLite pools have their own worker threads which are NOT
            // tokio tasks. Close them explicitly and wait for those threads as
            // well; otherwise the host shell keeps `sqlx-sqlite-worker` threads
            // alive after `zmodload -u`.
            history_db.close().await;
            history_store.store.close().await;
            // Also clears the cached data-dir / meta-store path so a later
            // Session re-resolves them instead of reusing this one's values.
            Settings::shutdown_process_state().await;
        });

        rt.shutdown_timeout(Duration::from_secs(10));
    }
}
