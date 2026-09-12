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
//!
//! Project-specific concerns that do not exist in the CLI (multi-thread
//! fire-and-forget `history end`, pre-`dlclose` resource shutdown, and the
//! custom terminal input that avoids signal-hook callbacks) are injected here
//! behind the `in-process` feature.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use atuin_client::database::{OptFilters, Sqlite, query_context};
use atuin_client::history::store::HistoryStore;
use atuin_client::history::{AuthorKind, AuthorPattern, History};
use atuin_client::record::sqlite_store::SqliteStore;
use atuin_client::settings::{FilterMode, RequestedSearchMode, Settings};
use atuin_common::encryption::paseto_v4;
use atuin_common::filter::OrFilter;
use eyre::Result;
use tokio::runtime::Runtime;
use tokio::sync::Notify;

use crate::command::client::history::{handle_end, handle_start_with_cwd};
use crate::command::client::search::interactive;
use crate::command::client::search::run_non_interactive_with_context;
use atuin_client::theme::ThemeManager;

/// Options for one non-interactive history search.
///
/// The field set mirrors the upstream `atuin search` command's filtering and
/// search-mode flags. A `None`/empty value means "use the configured default",
/// matching the CLI when the corresponding flag is omitted.
#[derive(Clone, Debug, Default)]
pub struct SearchOptions {
    /// Search text. Upstream treats multiple query words as a single
    /// space-joined query string for non-interactive searches.
    pub query: String,
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

/// A persistent in-process atuin session.
pub struct Session {
    settings: Arc<Settings>,
    history_db: Sqlite,
    history_store: HistoryStore,
    session_id: String,
    runtime: Option<Runtime>,
    in_flight_history_end: Arc<InFlightHistoryEnd>,
}

impl Session {
    /// Create a session.
    ///
    /// `Some(data_dir)` uses `data_dir` as the Atuin data directory.
    /// `None` follows the fully-resolved settings (`ATUIN_DATA_DIR`, XDG, or
    /// `data_dir` in config.toml) instead of second-guessing the config layer.
    pub fn new(data_dir: Option<&std::path::Path>) -> Result<Self> {
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
        })
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
        self.rt().block_on(handle_start_with_cwd(
            &self.history_db,
            &self.settings,
            command,
            cwd,
            author,
            author_kind,
            intent,
        ))
    }

    /// Blocking history end. Mirrors the official `atuin history end`
    /// implementation (duration is inferred from the start timestamp when no
    /// duration is supplied).
    pub fn history_end(&self, id: &str, exit: i64, duration_ns: u64) -> Result<()> {
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
                handle_end(&db, store, history_store, &settings, &id, exit, duration).await
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
        Ok(self.rt().block_on(self.history_db.load(id))?)
    }

    /// Search history using the same non-interactive options as the upstream
    /// `atuin search` command. Results are returned as history entries (newest
    /// first unless `reverse` is set).
    pub fn search(&self, options: SearchOptions) -> Result<Vec<History>> {
        self.rt().block_on(async {
            let mut search_settings = (*self.settings).clone();
            if let Some(mode) = options.search_mode {
                search_settings.requested_search_mode = mode;
            }
            if let Some(filter_mode) = options.filter_mode {
                search_settings.filter_mode = Some(filter_mode);
            }

            let authors = OrFilter::from_list(options.authors).unwrap_or_default();
            let shells = OrFilter::from_list(options.shells).unwrap_or_default();

            let context = query_context().await?;
            let opt_filter = OptFilters {
                exit: options.exit,
                exclude_exit: options.exclude_exit,
                only_failed: false,
                cwd: options.cwd.as_deref(),
                exclude_cwd: options.exclude_cwd.as_deref(),
                before: options.before.as_deref(),
                after: options.after.as_deref(),
                limit: options.limit,
                offset: options.offset,
                reverse: options.reverse,
                include_duplicates: options.include_duplicates,
                authors: authors.as_slice_filter(),
                shells: shells.as_slice_filter(),
            };

            let query = [options.query];
            run_non_interactive_with_context(
                &search_settings,
                opt_filter,
                &query,
                &self.history_db,
                context,
            )
            .await
        })
    }

    /// Convenience for the shell autosuggest path: official
    /// `atuin search --cmd-only --author '$all-user' --limit N --search-mode prefix`.
    pub fn search_prefix(&self, query: &str, limit: usize) -> Result<Vec<History>> {
        self.search(SearchOptions {
            query: query.to_owned(),
            search_mode: Some(RequestedSearchMode::Prefix),
            limit: Some(i64::try_from(limit).unwrap_or(i64::MAX)),
            authors: vec![AuthorPattern::AllUser],
            ..Default::default()
        })
    }

    /// Run the upstream interactive search TUI in-process.
    ///
    /// Returns `Ok(None)` when the caller should leave the shell buffer
    /// unchanged (cancel / return-original), and `Ok(Some(command))` for an
    /// accepted history entry or a returned query.
    pub fn interactive_search(&self, initial_query: &str) -> Result<Option<String>> {
        let db = self.history_db.clone();
        let history_store = self.history_store.clone();
        let settings = self.settings.clone();
        let result = self.rt().block_on(async {
            let mut theme_manager = ThemeManager::new(settings.theme.debug, None);
            let theme =
                theme_manager.load_theme(settings.theme.name.as_str(), settings.theme.max_depth);
            interactive::history(&[initial_query.to_owned()], &settings, db, &history_store, theme)
                .await
        })?;

        if result.is_empty() {
            Ok(None)
        } else {
            Ok(Some(result))
        }
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
            Settings::close_meta_store().await;
        });

        rt.shutdown_timeout(Duration::from_secs(10));
    }
}
