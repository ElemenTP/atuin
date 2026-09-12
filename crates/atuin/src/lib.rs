//! Library interface for the Atuin client.
//!
//! The binary target (`main.rs`) historically owned all command modules. To let
//! shell integrations embed Atuin in-process and reuse the exact command
//! implementations (history start/end, search, interactive TUI), this crate
//! also builds as a library. The `in-process` feature additionally exposes a
//! thin [`session`] wrapper around those command modules, following the same
//! pattern as zoxide's `src/session.rs`.

#[cfg(feature = "client")]
pub(crate) mod command;
pub(crate) mod logs;
#[cfg(feature = "client")]
pub(crate) mod shell;
#[cfg(feature = "sync")]
mod print_error;
#[cfg(feature = "sync")]
mod sync;

#[cfg(feature = "in-process")]
pub mod session;

const VERSION: &str = env!("CARGO_PKG_VERSION");
const SHA: &str = env!("GIT_HASH");
const LONG_VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), " (", env!("GIT_HASH"), ")");


// The CLI binary parses the same `Atuin` type. The library needs it too because
// the command tree (completion generation, external subcommands) is shared with
// the binary crate.
#[cfg(feature = "client")]
#[derive(clap::Parser)]
#[command(
    author = "Ellie Huxtable <ellie@atuin.sh>",
    version = VERSION,
    long_version = LONG_VERSION,
)]
pub(crate) struct Atuin {
    #[command(subcommand)]
    pub(crate) atuin: command::AtuinCmd,
}
