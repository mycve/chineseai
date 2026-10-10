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
    /// 冻结网络，校准已有策略因子并保存新模型。
    AzCalibratePolicy(AzCalibratePolicyArgs),
    /// Search one position and print policy/debug details.
    AzSearch(AzSearchArgs),
    /// Benchmark fixed-position search speed.
    AzBench(AzBenchArgs),
    /// Run self-play training from a TOML config.
    AzLoop(AzLoopArgs),
    /// Find "dive" positions by playing from the opening book against Pikafish.
    DiveGames(DiveGamesArgs),
    /// Run ChineseAI against a Pikafish UCI engine.
    VsPikafish(VsPikafishArgs),
}

#[derive(Args, Debug)]
#[command(after_long_help = "\
A frame is kept only when it is a \"blind spot\": the truth says we are already
lost while we have not noticed, and the two are far enough apart.

  pika_q <= --lost-below      (Pikafish: we are lost)
  our_q  >= --lost-below      (we have not noticed)
  |our_q - pika_q| >= --delta-q

Win rate is the metric because Pikafish `wdl` and our WDL head are the same
quantity. Only the FEN (plus the decision inputs) is stored: these positions are
fed back to reinforcement learning to explore, not labelled for supervision.
Endgames are excluded by --min-pieces / --max-dive-ply, and a game stops once
--max-frames-per-game frames have been collected.

Examples:
  chineseai dive-games ./tools/pikafish best.safetensors --games 500 --parallel-games 16
  chineseai dive-games ./tools/pikafish best.safetensors --games 200 \\
      --opening-book book.pgn.gz --opening-positions 2000 --pikafish-depth 12")]
pub(crate) struct DiveGamesArgs {
    /// Pikafish UCI executable path.
    pub(crate) pikafish_exe: String,
    /// ChineseAI AZ-NNUE model path.
    pub(crate) model: String,
    /// Output SQLite dive library.
    #[arg(long, default_value = "dive.sqlite")]
    pub(crate) output: String,
    /// Px0 opening book (book.pgn.gz) used to generate start positions.
    #[arg(long, default_value = "book.pgn.gz")]
    pub(crate) opening_book: String,
    /// Number of shuffled start positions taken from the opening book.
    #[arg(long, default_value_t = 500)]
    pub(crate) opening_positions: usize,
    /// Total games (ChineseAI plays Red in even games, Black in odd games).
    #[arg(long, default_value_t = 40)]
    pub(crate) games: usize,
    /// Simultaneous games / long-lived Pikafish processes.
    #[arg(long, default_value_t = 8)]
    pub(crate) parallel_games: usize,
    /// ChineseAI MCTS simulations per move.
    #[arg(short = 's', long, default_value_t = 800)]
    pub(crate) simulations: usize,
    /// Pikafish search depth during play (also the evaluation depth when --analyze-depth is 0).
    #[arg(long, default_value_t = 12)]
    pub(crate) pikafish_depth: u32,
    /// Pikafish depth used to re-evaluate dive candidates; 0 reuses --pikafish-depth.
    #[arg(long, default_value_t = 0)]
    pub(crate) analyze_depth: u32,
    /// Draw after this many plies.
    #[arg(long, default_value_t = 200)]
    pub(crate) max_plies: usize,
    /// Disagreement threshold on the win-rate scale: |our_q - pika_q| >= delta_q.
    /// Win rate is the primary metric because Pikafish wdl and our WDL head are
    /// the same quantity; cp is only recorded.
    #[arg(long, default_value_t = 0.35)]
    pub(crate) delta_q: f32,
    /// We count a position only when the truth says we are already lost
    /// (`pika_q <= lost_below`) while we have not noticed (`our_q >= lost_below`),
    /// with the two at least --delta-q apart.
    #[arg(long, default_value_t = -0.80, allow_negative_numbers = true)]
    pub(crate) lost_below: f32,
    /// Optional extra drop gate on the win-rate scale; 0 disables.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) drop_q: f32,
    /// Optional extra disagreement gate in centipawns; 0 disables.
    #[arg(long, default_value_t = 0)]
    pub(crate) delta_cp: i32,
    /// Optional extra drop gate in centipawns; 0 disables.
    #[arg(long, default_value_t = 0)]
    pub(crate) drop_cp: i32,
    /// Stop a game once this many dive frames have been collected. 0 disables.
    #[arg(long, default_value_t = 4)]
    pub(crate) max_frames_per_game: usize,
    /// Endgame exclusion: minimum total pieces on the board.
    #[arg(long, default_value_t = 20)]
    pub(crate) min_pieces: usize,
    /// Endgame exclusion: no dive is taken beyond this ply.
    #[arg(long, default_value_t = 80)]
    pub(crate) max_dive_ply: usize,
    /// Empty the dives table before collecting.
    #[arg(long)]
    pub(crate) clear: bool,
    /// Print the final FEN and move list of every game.
    #[arg(long)]
    pub(crate) report_games: bool,
    /// Pikafish Threads option.
    #[arg(long, default_value_t = 1)]
    pub(crate) pikafish_threads: u32,
    /// Pikafish Hash option in MB.
    #[arg(long, default_value_t = 64)]
    pub(crate) pikafish_hash_mb: u32,
    /// Pikafish NNUE file; empty lets the engine find pikafish.nnue next to the executable.
    #[arg(long, default_value = "")]
    pub(crate) pikafish_nnue: String,
    /// Allow the engine's own opening book during play (off by default so evals stay honest).
    #[arg(long)]
    pub(crate) pikafish_use_book: bool,
    /// Skip the first N games (resume a previous run: same --seed remaps the same
    /// opening position to the same game index, so skipping is exact).
    #[arg(long, default_value_t = 0)]
    pub(crate) skip_games: usize,
    /// Stop after N games this run; 0 runs all --games. Ctrl+C also stops cleanly
    /// after the current game finishes.
    #[arg(long, default_value_t = 0)]
    pub(crate) stop_after: usize,
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
    /// Random seed.
    #[arg(long, default_value_t = 20260411)]
    pub(crate) seed: u64,
}

#[derive(Args, Debug, Clone)]
pub(crate) struct AzInitArgs {
    /// Hidden size of the model.
    #[arg(default_value_t = chineseai::az::AzNnueArch::default().hidden_size)]
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

#[derive(Args, Debug)]
pub(crate) struct AzCalibratePolicyArgs {
    /// 输入模型路径。
    #[arg(long)]
    pub(crate) model: String,
    /// Px0 训练 TAR；固定留出组自动排除。
    #[arg(long)]
    pub(crate) train: String,
    /// 新模型路径，拒绝覆盖已有文件。
    #[arg(long)]
    pub(crate) output: String,
    /// 最多加载的完整训练对局数。
    #[arg(long, default_value_t = 512)]
    pub(crate) games: usize,
    /// 策略因子更新次数。
    #[arg(long, default_value_t = 1000)]
    pub(crate) steps: usize,
    #[arg(long, default_value_t = 17)]
    pub(crate) seed: u64,
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

    /// 打开根节点 check-only 连杀证明搜索，值为最大半回合数（0 = 关闭，15 = 最多 mate in 8 手）。
    #[arg(long, default_value_t = 0)]
    pub(crate) mate_search_plies: usize,
    /// 连杀证明的节点预算（用延迟换可证深度）：默认 20 万只够 mate-in-8。
    #[arg(long, default_value_t = 200_000)]
    pub(crate) mate_search_nodes: usize,
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
    /// Skip the first N games (resume a previous run with the same --seed).
    #[arg(long, default_value_t = 0)]
    pub(crate) skip_games: usize,
    /// Stop after N games this run; 0 means run all --games.
    #[arg(long, default_value_t = 0)]
    pub(crate) stop_after: usize,
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
    /// Px0 book.pgn.gz used for random start positions. Empty uses startpos.
    #[arg(long, default_value = "book.pgn.gz")]
    pub(crate) opening_book: String,
    /// Number of shuffled FEN positions to take from the opening book.
    #[arg(long, default_value_t = 1000)]
    pub(crate) opening_positions: usize,
    /// Pikafish Threads option.
    #[arg(long, default_value_t = 1)]
    pub(crate) pikafish_threads: u32,
    /// Pikafish Hash option in MB.
    #[arg(long, default_value_t = 64)]
    pub(crate) pikafish_hash_mb: u32,
    /// Pikafish NNUE file; empty lets the engine find pikafish.nnue next to the executable.
    #[arg(long, default_value = "")]
    pub(crate) pikafish_nnue: String,
    /// Allow the engine's own opening book during play.
    #[arg(long)]
    pub(crate) pikafish_use_book: bool,
}
