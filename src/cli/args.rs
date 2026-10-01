use crate::cli::az_loop_config::DEFAULT_AZ_LOOP_CONFIG;
use clap::{Args, Parser, Subcommand, ValueEnum};

const DEFAULT_VS_PIKAFISH_DEPTH: u32 = 10;
const DEFAULT_VS_PIKAFISH_GAMES: usize = 20;
const DEFAULT_VS_PIKAFISH_PARALLEL_GAMES: usize = 128;

#[derive(Parser, Debug)]
#[command(
    name = "chineseai",
    version,
    about = "ChineseAI AZ-NNUE search and training tools",
    long_about = "ChineseAI AZ-NNUE search and training tools."
)]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Option<CliCommand>,
}

#[derive(Subcommand, Debug)]
pub(crate) enum CliCommand {
    /// Create a random AZ-NNUE model.
    AzInit(AzInitArgs),
    /// Scale one policy component for structural ablation.
    AzPolicyScale(AzPolicyScaleArgs),
    /// Search one position and print policy/debug details.
    AzSearch(AzSearchArgs),
    /// Benchmark fixed-position search speed.
    AzBench(AzBenchArgs),
    /// Run self-play training from a TOML config.
    AzLoop(AzLoopArgs),
    /// Run ChineseAI against a Pikafish UCI engine.
    VsPikafish(VsPikafishArgs),
}

#[derive(Args, Debug, Clone)]
pub(crate) struct AzInitArgs {
    /// Hidden size of the model.
    #[arg(default_value_t = 128)]
    pub(crate) hidden: usize,
    /// Output model path.
    #[arg(default_value = "model.safetensors")]
    pub(crate) output: String,
    /// Random seed.
    #[arg(default_value_t = 20260409)]
    pub(crate) seed: u64,
}

impl AzInitArgs {
    pub(crate) fn arch(&self) -> chineseai::az::AzNnueArch {
        chineseai::az::AzNnueArch::with_hidden_size(self.hidden.max(1))
    }
}

#[derive(Args, Debug, Clone)]
pub(crate) struct AzPolicyScaleArgs {
    /// Existing model path.
    pub(crate) input: String,
    /// Modified model path.
    pub(crate) output: String,
    /// Policy component to scale.
    #[arg(long, value_enum)]
    pub(crate) component: PolicyComponent,
    /// Multiplier applied to the selected component.
    #[arg(long)]
    pub(crate) scale: f32,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub(crate) enum PolicyComponent {
    Exact,
    Capture,
    Factor,
    Tactical,
    TacticalExact,
    TacticalFactor,
    Accumulator,
    Context,
    Consequence,
    MoveBias,
    ThreatContext,
}

#[derive(Args, Debug, Clone)]
#[command(after_long_help = "\
Examples:
  chineseai az-search model.safetensors
  chineseai az-search model.safetensors 50000 1.5 --top 12 startpos
  chineseai az-search model.safetensors 10000 --trace-move b0c2 --verify-top 3 startpos")]
pub(crate) struct AzSearchArgs {
    /// AZ-NNUE model path.
    pub(crate) model: String,
    /// Number of MCTS simulations.
    #[arg(default_value_t = 800)]
    pub(crate) simulations: usize,
    /// Non-root PUCT init.
    #[arg(default_value_t = 1.0)]
    pub(crate) cpuct: f32,
    /// Root PUCT init.
    #[arg(long, default_value_t = 1.9)]
    pub(crate) cpuct_at_root: f32,
    /// Non-root first-play urgency reduction.
    #[arg(long, default_value_t = 0.23)]
    pub(crate) fpu_value: f32,
    /// Root absolute first-play urgency.
    #[arg(long, default_value_t = 1.0)]
    pub(crate) fpu_value_at_root: f32,
    /// Divisor applied to policy logits before root search; above 1 flattens priors.
    #[arg(long, default_value_t = 1.4)]
    pub(crate) policy_softmax_temp: f32,
    /// Dynamic PUCT base.
    #[arg(long, default_value_t = 38739.0)]
    pub(crate) cpuct_base: f32,
    /// Dynamic PUCT growth factor.
    #[arg(long, default_value_t = 3.894)]
    pub(crate) cpuct_factor: f32,
    /// Root dynamic PUCT base.
    #[arg(long, default_value_t = 38739.0)]
    pub(crate) cpuct_base_at_root: f32,
    /// Root dynamic PUCT growth factor.
    #[arg(long, default_value_t = 3.894)]
    pub(crate) cpuct_factor_at_root: f32,
    /// Maximum search depth in plies below root; 0 keeps the MCTX default (simulations).
    #[arg(long, default_value_t = 0)]
    pub(crate) max_depth: usize,
    /// Draw value in Q = W - L + draw_score * D.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) draw_score: f32,
    /// Scale non-terminal network values during search; 0 isolates policy priors.
    #[arg(long, default_value_t = 1.0)]
    pub(crate) value_scale: f32,
    /// Independently re-search this many top-visited root moves after making each move.
    #[arg(long, default_value_t = 0)]
    pub(crate) verify_top: usize,
    /// Independently re-search specific root moves (repeat the option for multiple moves).
    #[arg(long = "verify-move")]
    pub(crate) verify_moves: Vec<String>,
    /// Restrict the root search to these legal moves (repeat for multiple moves).
    #[arg(long = "root-move")]
    pub(crate) root_moves: Vec<String>,
    /// Print the most-visited continuation below this root move with network leaf values.
    #[arg(long = "trace-move")]
    pub(crate) trace_move: Option<String>,
    /// Simulations for every independent child verification; 0 uses the root simulation count.
    #[arg(long, default_value_t = 0)]
    pub(crate) verify_sims: usize,
    /// Candidate rows to display, sorted by visits; 0 displays every legal root move.
    #[arg(long, default_value_t = 20)]
    pub(crate) top: usize,
    /// Apply legal UCI moves before searching; repeat for a move sequence.
    #[arg(long = "move")]
    pub(crate) moves: Vec<String>,
    /// FEN string, or startpos if omitted.
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    pub(crate) fen: Vec<String>,
}

#[derive(Args, Debug)]
#[command(after_long_help = "\
Examples:
  chineseai az-bench model.safetensors 512 100 1.5 startpos
  chineseai az-bench model.safetensors 512 100 1.5 startpos")]
pub(crate) struct AzBenchArgs {
    /// AZ-NNUE model path.
    pub(crate) model: String,
    /// Simulations per search.
    #[arg(default_value_t = 800)]
    pub(crate) simulations: usize,
    /// Number of repeated searches.
    #[arg(default_value_t = 100)]
    pub(crate) repeat: usize,
    /// PUCT constant for AlphaZero search.
    #[arg(default_value_t = 1.0)]
    pub(crate) cpuct: f32,
    /// FEN string, or startpos if omitted.
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    pub(crate) fen: Vec<String>,
}

#[derive(Args, Debug)]
pub(crate) struct AzLoopArgs {
    /// Training config path.
    #[arg(default_value = DEFAULT_AZ_LOOP_CONFIG)]
    pub(crate) config: String,
    /// Stop after completing this absolute update number and save the model/progress.
    #[arg(long)]
    pub(crate) target_update: Option<usize>,
}

#[derive(Args, Debug)]
#[command(after_long_help = "\
Examples:
  chineseai vs-pikafish ./tools/pikafish model.safetensors
  chineseai vs-pikafish ./tools/pikafish checkpoints/update-0620-model.safetensors --simulations 192
  chineseai vs-pikafish ./tools/pikafish model.safetensors --pikafish-depth 10 --games 40 --parallel-games 5
  chineseai vs-pikafish ./tools/pikafish model.safetensors --opening-book book.pgn.gz")]
pub(crate) struct VsPikafishArgs {
    /// Pikafish UCI executable path.
    pub(crate) pikafish_exe: String,
    /// ChineseAI AZ-NNUE model path.
    pub(crate) model: String,
    /// ChineseAI MCTS simulations per move.
    #[arg(short = 's', long, default_value = "800")]
    pub(crate) simulations: Option<usize>,
    /// ChineseAI PUCT constant.
    #[arg(long, default_value_t = 1.0)]
    pub(crate) cpuct: f32,
    /// ChineseAI root PUCT constant.
    #[arg(long, default_value_t = 1.9)]
    pub(crate) cpuct_at_root: f32,
    /// ChineseAI dynamic PUCT base.
    #[arg(long, default_value_t = 38739.0)]
    pub(crate) cpuct_base: f32,
    /// ChineseAI dynamic PUCT growth factor.
    #[arg(long, default_value_t = 3.894)]
    pub(crate) cpuct_factor: f32,
    /// ChineseAI root dynamic PUCT base.
    #[arg(long, default_value_t = 38739.0)]
    pub(crate) cpuct_base_at_root: f32,
    /// ChineseAI root dynamic PUCT growth factor.
    #[arg(long, default_value_t = 3.894)]
    pub(crate) cpuct_factor_at_root: f32,
    /// ChineseAI non-root first-play urgency reduction.
    #[arg(long, default_value_t = 0.23)]
    pub(crate) fpu_value: f32,
    /// ChineseAI root first-play urgency reduction.
    #[arg(long, default_value_t = 1.0)]
    pub(crate) fpu_value_at_root: f32,
    /// Divisor applied to ChineseAI policy logits before search.
    #[arg(long, default_value_t = 1.4)]
    pub(crate) policy_softmax_temp: f32,
    /// Draw after this many plies.
    #[arg(long, default_value_t = 200)]
    pub(crate) max_plies: usize,
    /// Random seed.
    #[arg(long, default_value_t = 20260411)]
    pub(crate) seed: u64,
    /// Pikafish search depth.
    #[arg(long, default_value_t = DEFAULT_VS_PIKAFISH_DEPTH)]
    pub(crate) pikafish_depth: u32,
    /// Total games.
    #[arg(long, default_value_t = DEFAULT_VS_PIKAFISH_GAMES)]
    pub(crate) games: usize,
    /// Simultaneous games/processes.
    #[arg(long, default_value_t = DEFAULT_VS_PIKAFISH_PARALLEL_GAMES)]
    pub(crate) parallel_games: usize,
    /// Print the final FEN and complete move list for every game.
    #[arg(long)]
    pub(crate) report_games: bool,
    /// Px0 book.pgn.gz used to generate random start positions. Empty uses startpos.
    #[arg(long, default_value = "book.pgn.gz")]
    pub(crate) opening_book: String,
    /// Number of shuffled FEN positions to take from the Px0 book.
    #[arg(long, default_value_t = 1000)]
    pub(crate) opening_positions: usize,
}
